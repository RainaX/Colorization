import time
from options.train_options import TrainOptions
from models import create_model

import torch
import torchvision
import torchvision.transforms as transforms

from util import util

if __name__ == '__main__':
    opt = TrainOptions().parse()

    opt.dataroot = '../11-785hw2p2-s20/train_data/%s/' % opt.phase
    dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.RandomChoice([transforms.Resize(opt.loadSize, interpolation=1),
                                                                            transforms.Resize(opt.loadSize, interpolation=2),
                                                                            transforms.Resize(opt.loadSize, interpolation=3),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=1),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=2),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=3)]),
                                                   transforms.RandomChoice([transforms.RandomResizedCrop(opt.fineSize, interpolation=1),
                                                                            transforms.RandomResizedCrop(opt.fineSize, interpolation=2),
                                                                            transforms.RandomResizedCrop(opt.fineSize, interpolation=3)]),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor()]))
                                                   # transforms.RandomChoice([transforms.ColorJitter(brightness=.05, contrast=.05, saturation=.05, hue=.05),
                                                   #                          transforms.ColorJitter(brightness=0, contrast=0, saturation=.05, hue=.1),
                                                   #                          transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0), ]),
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    model.print_networks(True)

    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # for i, data in enumerate(dataset):
        for i, data_raw in enumerate(dataset_loader):
            data_raw[0] = data_raw[0].cuda()
            data = util.get_colorization_data(data_raw, opt, p=opt.sample_p)
            if(data is None):
                continue

            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                # time to load data
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()


            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                # time to do forward&backward
                # t = (time.time() - iter_start_time) / opt.batch_size
                print(losses)
                t = time.time() - iter_start_time

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
