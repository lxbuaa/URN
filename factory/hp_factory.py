
def get_hp(hp, args):
    if args.train_set is not None:
        hp.data.train = args.train_set
    if args.test_set is not None:
        hp.data.test = args.test_set
    return hp
