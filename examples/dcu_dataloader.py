#!/usr/bin/env python3
"""Train a tiny next-frame model on a Falcons-compressed HDF5 dataset."""

import argparse
import math
from pathlib import Path
import sys
import tempfile


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="build a temporary synthetic dataset")
    source.add_argument("--input", type=Path, help="existing Falcons-compressed HDF5 file")
    parser.add_argument("--library", required=True, type=Path, help="native HIP codec shared library")
    parser.add_argument("--dataset", default="fields", help="HDF5 dataset path (default: fields)")
    parser.add_argument("--device", default="cuda:0", help="PyTorch HIP device (default: cuda:0)")
    parser.add_argument("--route", choices=("cpu_stream", "dcu", "dcu_stream"), default="dcu_stream",
                        help="one decode route for all workers (default: dcu_stream)")
    parser.add_argument("--num-workers", type=positive_int, default=2, help="fork workers (default: 2)")
    parser.add_argument("--reader-processes", type=nonnegative_int, default=1,
                        help="total CPU readers including the worker (default: 1; 0 aliases 1)")
    parser.add_argument("--batch-size", type=positive_int, default=2, help="windows per batch (default: 2)")
    parser.add_argument("--epochs", type=positive_int, default=2, help="number of epochs (default: 2)")
    parser.add_argument("--steps", type=positive_int, default=4, help="maximum batches per epoch (default: 4)")
    parser.add_argument("--channels", type=int, nargs="+", help="source channel indices in output order")
    parser.add_argument("--shuffle", action="store_true", help="shuffle window starts each epoch")
    args = parser.parse_args()

    if sys.platform != "linux":
        parser.error("this example requires Linux fork workers and a HIP-enabled PyTorch build")
    if not args.dataset:
        parser.error("--dataset must not be empty")
    if args.channels is not None and (
        any(channel < 0 for channel in args.channels) or len(set(args.channels)) != len(args.channels)
    ):
        parser.error("--channels must contain unique, nonnegative indices")
    args.library = args.library.expanduser().resolve()
    if not args.library.is_file():
        parser.error(f"native HIP library not found: {args.library}")
    cpu_library = args.library.parent / "libfalcons_loader_cpu.so"
    if not cpu_library.is_file():
        parser.error(f"CPU codec library not found: {cpu_library}")
    if args.input is not None:
        args.input = args.input.expanduser().resolve()
        if not args.input.is_file():
            parser.error(f"input file not found: {args.input}")
    else:
        if sys.byteorder != "little":
            parser.error("--demo requires a little-endian host")
        if args.dataset != "fields":
            parser.error("--demo creates only the 'fields' dataset")
        if args.batch_size > 16 or args.steps > 4096:
            parser.error("--demo requires --batch-size <= 16 and --steps <= 4096")
        if args.channels is not None and any(channel >= 4 for channel in args.channels):
            parser.error("--demo has four channels, indexed 0 through 3")
    return args


def train(args, path):
    import torch
    from hipfalcons import ERA5Loader

    loader = ERA5Loader(
        path,
        library_path=args.library,
        route=args.route,
        device=args.device,
        num_workers=args.num_workers,
        reader_processes=args.reader_processes,
        batch_size=args.batch_size,
        history=1,
        future=1,
        dataset=args.dataset,
        channels=args.channels,
        shuffle=args.shuffle,
        drop_last=False,
    )
    try:
        # This forks the persistent workers before any HIP/model initialization.
        loader.start()
        if torch.version.hip is None or not torch.cuda.is_available():
            raise RuntimeError("a HIP-enabled PyTorch build and an available DCU are required")

        device = torch.device(args.device)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(loader.channel_count, 8, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, loader.channel_count, kernel_size=1),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()
        model.train()
        print(f"route={args.route} reader_processes={loader.reader_processes} shuffle={args.shuffle} "
              f"shape={loader.shape} channels={loader.channels} device={device}", flush=True)

        for epoch in range(1, args.epochs + 1):
            iterator = iter(loader)
            try:
                for step, (history, future) in enumerate(iterator, start=1):
                    # Each tensor is already on the requested device: [B, 1, C, H, W].
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(history[:, 0])
                    loss = loss_fn(prediction, future[:, 0])
                    loss_value = loss.item()
                    if not math.isfinite(loss_value):
                        raise RuntimeError("non-finite loss; check input values, scaling, and learning rate")
                    loss.backward()
                    optimizer.step()
                    print(f"epoch={epoch} step={step} batch={history.shape[0]} loss={loss_value:.6g}", flush=True)
                    if step >= args.steps:
                        break
            finally:
                # A step limit or an exception may leave prefetched batches behind.
                iterator.close()
    finally:
        loader.close()


def main():
    args = parse_args()
    if args.input is not None:
        train(args, args.input)
        return

    from hipfalcons.bench import build_fixture

    # The benchmark builder creates batches + 1 batches and an on-disk CPU oracle.
    fixture_args = argparse.Namespace(
        library=str(args.library),
        encoder_library=None,
        shape=(96, 120),
        batch_size=args.batch_size,
        batches=args.steps,
        limit_mib=128,
        abs_error=1e-3,
        seed=20261119,
        timeout=600.0,
    )
    with tempfile.TemporaryDirectory(prefix="hipfalcons-example-") as directory:
        path, _oracle, details = build_fixture(fixture_args, Path(directory))
        print(f"demo fixture shape={details['shape']} abs_bound={details['abs_bound']}", flush=True)
        train(args, path)


if __name__ == "__main__":
    main()
