import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog='rickroll', description='Ordnet 1-Sekunden-Schnipsel einem Song zu')
    sub = ap.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train', help='Modell auf den Songs trainieren')
    t.add_argument('--songs', default='songs')
    t.add_argument('--out', default='model.pt')
    t.add_argument('--epochs', type=int, default=15)
    t.add_argument('--steps', type=int, default=2000, help='Batches pro Epoche')

    p = sub.add_parser('predict', help='Sekunden einer .wav-Datei zuordnen')
    p.add_argument('wav')
    p.add_argument('--model', default='model.pt')
    p.add_argument('--second', type=int, default=None)

    args = ap.parse_args()
    if args.cmd == 'train':
        from .train import train
        train(args.songs, args.out, args.epochs, steps_per_epoch=args.steps)
    else:
        from .predict import predict
        predict(args.wav, args.model, args.second)
