from train_base import *

# constants
SYNC = False
GET_MODULE = False

def main():
    args = parse_args()

    # Init dist
    local_rank = 0
    global_rank = 0
    world_size = 1

    args = init_env(args, local_rank, global_rank)

    model = init_models(args)
    
    train_sampler, dataloader = init_dataset(args, global_rank, world_size)
    val_sampler, val_dataloader = init_dataset(args, global_rank, world_size, True)

    model = load_dicts(args, GET_MODULE, model)

    optimizer = init_optims(args, world_size, model)

    lr_scheduler = init_schedulers(args, dataloader, optimizer)

    train(args, global_rank, SYNC, GET_MODULE,
            model,
            train_sampler, dataloader, val_sampler, val_dataloader,
            optimizer,
            lr_scheduler)

if __name__ == '__main__':
    main()