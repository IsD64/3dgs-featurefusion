from gaussian_splatting.prepare import prepare_dataset
from feature_3dgs.dataset import build_dataset


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-n", "--name", required=True, type=str)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    args = parser.parse_args()
    dataset = prepare_dataset(source=args.source, device=args.device, trainable_camera=False, load_camera=None, load_mask=False, load_depth=False)

    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    dataset = build_dataset(name=args.name, cameras=dataset, **configs)
    # TODO: PCA and save the dataset to args.destination
