import os, time, glob
import logging
import argparse
import numpy as np

# import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

# from model_search import Network, Architecture
# from model_search_pcdarts import Network, Architecture
import utils
from resnet import *
import copy
import math
import unittest
import sys
import torch.nn as nn
import coco_data_loader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from model_search import Network, Architecture

# from architect_ts import Architect
import json
from student import *

# from student_update import *
# from build_vocab import *
import pickle

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


parser = argparse.ArgumentParser("coco_caption")
parser.add_argument(
    "--data", type=str, default="../data", help="location of the data corpus"
)
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.025, help="init learning rate"
)
parser.add_argument(
    "--learning_rate_min", type=float, default=0.001, help="min learning rate"
)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--epochs", type=int, default=50, help="num of training epochs")
parser.add_argument(
    "--init_channels", type=int, default=16, help="num of init channels"
)
parser.add_argument("--layers", type=int, default=8, help="total number of layers")
parser.add_argument(
    "--model_path", type=str, default="saved_models", help="path to save the model"
)
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--debug", action="store_true", default=False, help="Debug")
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
parser.add_argument(
    "--drop_path_prob", type=float, default=0.3, help="drop path probability"
)
parser.add_argument("--save", type=str, default="EXP", help="experiment name")
parser.add_argument("--seed", type=int, default=2, help="random seed")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument(
    "--train_portion", type=float, default=0.5, help="portion of training data"
)
parser.add_argument(
    "--unrolled",
    action="store_true",
    default=False,
    help="use one-step unrolled validation loss",
)
parser.add_argument(
    "--arch_learning_rate",
    type=float,
    default=3e-4,
    help="learning rate for arch encoding",
)
parser.add_argument(
    "--arch_weight_decay",
    type=float,
    default=1e-3,
    help="weight decay for arch encoding",
)
parser.add_argument("--arch_steps", type=int, default=4, help="architecture steps")
parser.add_argument("--unroll_steps", type=int, default=1, help="unrolling steps")
parser.add_argument("--lam", type=float, help="lambda", default=1)
parser.add_argument("--gamma", type=float, help="gamma", default=1)
parser.add_argument("--enc_dec_learning_rate", type=float, default=1e-3)
parser.add_argument("--enc_dec_weight_decay", type=float, default=1e-3)
parser.add_argument("--learner_learning_rate", type=float, default=1e-3)
parser.add_argument("--learner_weight_decay", type=float, default=1e-3)
parser.add_argument("--is_parallel", type=int, default=0)
parser.add_argument("--student_arch", type=str, default="18")
parser.add_argument("--lang_score_freq", type=int, default=3)
parser.add_argument("--darts_type", type=str, default="DARTS", help="[DARTS, PCDARTS]")


args = parser.parse_args()

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log_iuc.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("gpu device = %d" % args.gpu)
logging.info("args = %s", args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:0")

args.save = "search-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

json_file_path = os.path.join(args.data, "cocotalk.json")
h5_file_path = os.path.join(args.data, "cocotalk.h5")


CIFAR_CLASSES = 10
CIFAR100_CLASSES = 100


def idx_2_words(idxs, vocab):
    ans = ""
    for idx in idxs.squeeze().cpu().numpy():
        # print(idx.shape)
        # print(vocab.keys())
        if ans == "" and int(idx) != 0:
            ans += vocab[str(int(idx))]
        elif int(idx) != 0:
            ans += " " + vocab[str(int(idx))]
    return ans


# if args.darts_type == 'DARTS':
#     from model_search import Network, Architecture
# elif args.darts_type == 'PCDARTS':
#     from model_search_pcdarts import Network, Architecture


np.random.seed(args.seed)
if not args.is_parallel:
    torch.cuda.set_device(int(args.gpu))
    logging.info("gpu device = %d" % int(args.gpu))
else:
    logging.info("gpu device = %s" % args.gpu)
cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled = True
torch.cuda.manual_seed(args.seed)
logging.info("args = %s", args)

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

transforms = utils._data_transforms_coco(args)  # change for coco
vocab = json.load(open(json_file_path))["ix_to_word"]
train_queue, valid_queue, external_queue = coco_data_loader.get_loader(
    json_file=json_file_path,
    h5_file=h5_file_path,
    transform=transforms,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2,
    args=args,
    debug=args.debug,
)


decoder1 = RNNDecoder(vocab_size=len(vocab) + 1, hidden_size=1024).cuda()
decoder2 = RNNDecoder(vocab_size=len(vocab) + 1, hidden_size=1024).cuda()
learner = Learner(
    enc_arch=args.student_arch, vocab_size=len(vocab) + 1, decoder=decoder1
).cuda()
model = Network(
    C=args.init_channels,
    layers=args.layers,
    criterion=criterion,
    decoder=decoder2,
    steps=4,
    multiplier=4,
    stem_multiplier=3,
).cuda()
# model = Network(args.init_channels, args.layers, criterion, decoder=decoder2).cuda()


num_train = len(train_queue)  # 50000
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))
report_freq = int(num_train * args.train_portion // args.batch_size + 1)
train_iters = int(
    args.epochs
    * (num_train * args.train_portion // args.batch_size + 1)
    * args.unroll_steps
)


class Outer(ImplicitProblem):
    def forward(self):
        return self.module()

    def training_step(self, batch):
        input, caption, length, info = batch

        alphas = self.forward()
        input = input.cuda()
        caption = caption.cuda()
        target = caption.detach().cpu()
        logits = self.inner1.module(input, caption, length)
        loss = self.inner1.module.loss(input, caption, length).detach()
        n = input.size(0)

        latent_feats = self.inner1.module.encode(input)
        idxs = self.inner1.module._decoder.sample(latent_feats).detach()
        for i, idxz in enumerate(idxs):
            pred = idx_2_words(idxz, vocab)

            preds.append({"image_id": info[i]["id"], "caption": pred})

        #         prec1, prec5 = utils.accuracy(logits.cpu(), target.cpu(), topk=(1, 5))
        #         objs.update(loss.item(), n)
        #         top1.update(prec1.item(), n)
        #         top5.update(prec5.item(), n)

        #         if step % args.report_freq == 0:
        #             logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        #         lang_scores = utils.score_lang(preds, args, cider_only=cider_only)
        #         # print(lang_scores)

        assert not math.isnan(loss)
        epoch = int(
            self.count
            * (args.batch_size + 1)
            * args.unroll_steps
            // (num_train * args.train_portion)
        )
        print(f"Epoch: {epoch} || step: {self.count} || loss: {loss.item()}")

        return loss

    def configure_train_data_loader(self):
        train_queue, valid_queue, external_queue = coco_data_loader.get_loader(
            json_file=json_file_path,
            h5_file=h5_file_path,
            transform=transforms,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            args=args,
            debug=args.debug,
        )
        return valid_queue

    def configure_module(self):
        return Architecture(steps=args.arch_steps).to(device)

    def configure_optimizer(self):
        optimizer = optim.Adam(
            self.module.parameters(),
            lr=args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.learner_weight_decay,
        )
        return optimizer


class Inner2(ImplicitProblem):
    def forward(self, input, alphas, captions, lengths):
        return self.module(input, alphas, captions, lengths)

    def training_step(self, batch):
        input_external, captions_external, lengths_external, infos_external = batch
        input_external = input_external.cuda()
        captions_external = captions_external.cuda(non_blocking=True)

        # make pseudo dataset using enc-dec
        (
            input_pseudo,
            captions_pseudo,
            lengths_pseudo,
        ) = coco_data_loader.get_pseudo_loader(self.inner1.module, input_external)
        # train learner using pseudo dataset (train)

        #############################################################################################
        alphas = self.outer()
        loss = self.module.loss(input_pseudo, alphas, captions_pseudo, lengths_pseudo)

        #############################################################################################

        return loss

    def configure_train_data_loader(self):
        train_queue, valid_queue, external_queue = coco_data_loader.get_loader(
            json_file=json_file_path,
            h5_file=h5_file_path,
            transform=transforms,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            args=args,
            debug=args.debug,
        )
        return external_queue

    def configure_module(self):
        return learner

    def configure_optimizer(self):
        optimizer = torch.optim.SGD(
            self.module.parameters(),
            args.learner_learning_rate,
            momentum=args.momentum,
            weight_decay=args.learner_weight_decay,
        )
        return optimizer

    def configure_scheduler(self):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(args.epochs), eta_min=args.learning_rate_min
        )
        return scheduler


class Inner1(ImplicitProblem):
    def forward(self, input, alphas, captions, lengths):
        return self.module(input, alphas, captions, lengths)

    def training_step(self, batch):
        input, captions, lengths, infos = batch
        n = input.size(0)
        input = input.cuda()
        captions = captions.cuda(non_blocking=True)
        #############################################################################################
        alphas = self.outer()
        target = captions.detach().cpu()
        logits = self.module(input, alphas, captions, lengths).detach().cpu()
        loss = self.module.loss(input, alphas, captions, lengths)
        #############################################################################################

        if args.debug and step % 5 == 0:
            # print(loss.item())
            temp_logits = logits.view((captions.shape[0], captions.shape[1], -1))
            for batch in range(min(temp_logits.shape[0], 3)):
                idxz = torch.argmax(temp_logits[batch, :, :], 1).detach()
                print("train pred: ", idx_2_words(idxz, vocab))
                print("real: ", idx_2_words(captions[batch, :], vocab))

        return loss

    def configure_train_data_loader(self):
        train_queue, valid_queue, external_queue = coco_data_loader.get_loader(
            json_file=json_file_path,
            h5_file=h5_file_path,
            transform=transforms,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            args=args,
            debug=args.debug,
        )
        return train_queue

    def configure_module(self):
        return model
        # return Network(args.init_channels, args.layers, criterion).cuda()

    def configure_optimizer(self):
        optimizer = torch.optim.SGD(
            self.module.parameters(),
            args.enc_dec_learning_rate,
            momentum=args.momentum,
            weight_decay=args.enc_dec_weight_decay,
        )
        return optimizer

    def configure_scheduler(self):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(args.epochs), eta_min=args.learning_rate_min
        )
        return scheduler


class NASEngine(Engine):
    @torch.no_grad()
    def validation(self):
        #         corrects = 0
        #         total = 0
        #         for x, target in test_queue:
        #             x, target = x.to(device), target.to(device, non_blocking=True)
        #             alphas = self.outer()
        #             _, correct = self.inner1.module.loss(x, alphas, target, acc=True)
        #             corrects += correct
        #             total += x.size(0)

        #         acc = corrects / total
        #         logging.info('[*] Valid Acc.: %f', acc)
        #         print("[*] Valid Acc.:", acc)

        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()
        preds = []
        with torch.no_grad():
            for step, (input, caption, length, info) in enumerate(valid_queue):
                input = input.cuda()
                alphas = self.outer()
                caption = caption.cuda()
                target = caption.detach().cpu()
                logits = self.inner1.module(input, alphas, caption, length)
                loss = self.inner1.module.loss(input, alphas, caption, length).detach()
                n = input.size(0)

                latent_feats = self.inner1.module.encode(input)
                idxs = self.inner1.module._decoder.sample(latent_feats).detach()
                for i, idxz in enumerate(idxs):
                    pred = idx_2_words(idxz, vocab)

                    preds.append({"image_id": info[i]["id"], "caption": pred})

                prec1, prec5 = utils.accuracy(logits.cpu(), target.cpu(), topk=(1, 5))
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % args.report_freq == 0:
                    logging.info(
                        "valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg
                    )

        # lang_scores = utils.score_lang(preds, args, cider_only=cider_only)

        alphas = self.outer()
        logging.info("genotype = %s", self.inner1.module.genotype(alphas))
        torch.save({"genotype": self.inner1.module.genotype(alphas)}, "genotype.t7")


# outer_config = Config(retain_graph=True, first_order=True,log_step=1, fp16=True)
# inner2_config = Config(type="darts", unroll_steps=args.unroll_steps, allow_unused=True, fp16=True)
# inner1_config = Config(type="darts", unroll_steps=args.unroll_steps, allow_unused=True, fp16=True)

outer_config = Config(retain_graph=True, first_order=True, log_step=1)
inner2_config = Config(type="darts", unroll_steps=args.unroll_steps, allow_unused=True)
inner1_config = Config(type="darts", unroll_steps=args.unroll_steps, allow_unused=True)
engine_config = EngineConfig(
    valid_step=report_freq,
    train_iters=train_iters,
    roll_back=True,
)
outer = Outer(name="outer", config=outer_config, device=device)
inner1 = Inner1(name="inner1", config=inner1_config, device=device)
inner2 = Inner2(name="inner2", config=inner2_config, device=device)


problems = [outer, inner2, inner1]
# l2u = {inner1: [outer], inner1: [inner2],  inner2: [outer], inner1: [inner2,outer]}
l2u = {inner1: [inner2, outer], inner2: [outer]}
u2l = {outer: [inner2, inner1]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = NASEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()
