import horovod.torch as hvd

from train_base import *

# constants
SYNC = True
GET_MODULE = False

def main():
    args = parse_args()

    # Init horovod
    hvd.init()

    local_rank = hvd.local_rank()
    global_rank = hvd.rank()
    world_size = hvd.size()

    args = init_env(args, local_rank, global_rank)

    model = init_models(args)
    
    train_sampler, dataloader = init_dataset(args, global_rank, world_size)
    val_sampler, val_dataloader = init_dataset(args, global_rank, world_size, True)

    model = load_dicts(args, GET_MODULE, model)

    optimizer = init_optims(args, world_size, model)

    # Wrap the optimizers
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    lr_scheduler = init_schedulers(args, dataloader, optimizer)

    # Boardcast
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    train(args, global_rank, SYNC, GET_MODULE,
            model,
            train_sampler, dataloader, val_sampler, val_dataloader,
            optimizer,
            lr_scheduler)

if __name__ == '__main__':
    main()