"""
Experiment class for unified model
"""

import config
import torch
from experiment import Experiment
from torch import nn
from misc import unified_ans_acc, calc_bleu_scores_unified
from itertools import cycle


class ExperimentUnified(Experiment):
    def __init__(self, args):
        super(ExperimentUnified, self).__init__(args)
        self.unified_vocab = self.data_loader["train"].dataset.dataset.unified_vocab

    def evaluate_gen_qa(self, batch_sample):
        """
        Helper routine to evaluate generated qst+ans
        """
        self.vqa_model.eval()
        # import pdb; pdb.set_trace()
        image = batch_sample["image"].to(config.DEVICE)
        qa_str = batch_sample["qa_str"]
        image_path = batch_sample["image_path"]
        # ground truth question+answer
        qa_str = [self.unified_vocab.arr2qst(qa) for qa in qa_str]

        # generated question-answer
        with torch.no_grad():
            gen_qa_str = self.vqa_model.generate(image)
        gen_qa_str = [self.unified_vocab.arr2qst(qa) for qa in gen_qa_str]

        n = min(4, len(image))
        self.log("Evaluating question answer pairs")
        for i in range(n):
            self.log(f"image path:{image_path[i]}")
            self.log(f"ground truth qa: {qa_str[i]}")
            self.log(f"generated qa: {gen_qa_str[i]}")

    def train(self):
        self.vqa_model.train()
        total_loss = 0
        total_ans_acc = 0
        num_batches = len(self.data_loader["train"])
        valid_queue_iter = cycle(iter(self.data_loader["valid"]))
        lr = self.scheduler.get_lr()[0]
        # import pdb; pdb.set_trace()

        for batch_idx, batch_sample in enumerate(self.data_loader["train"]):
            # get training data
            image = batch_sample["image"].to(config.DEVICE)
            qa_str = batch_sample["qa_str"].to(config.DEVICE)

            # STAGE1: architecture update
            if self.arch_type == "darts" and (batch_idx % self.arch_update_freq == 0):
                batch_sample = next(valid_queue_iter)
                val_image = batch_sample["image"].to(config.DEVICE)
                val_qa_str = batch_sample["qa_str"].to(config.DEVICE)
                label, val_label = None, None
                # import pdb; pdb.set_trace()
                self.architect.step(
                    image, qa_str, label, val_image, val_qa_str, val_label, lr
                )

            # train model
            self.optimizer.zero_grad()
            qa_out = self.vqa_model(image, qa_str)
            qa_flat = qa_str[:, 1:].flatten()
            qa_pred_flat = qa_out[:, :-1].flatten(end_dim=1)
            loss = self.criterion(qa_pred_flat, qa_flat)
            loss.backward()
            nn.utils.clip_grad_norm_(self.vqa_model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()

            # calculate accuracy
            qa_pred = torch.argmax(qa_out, dim=2)
            ans_acc = unified_ans_acc(qa_str, qa_pred, self.unified_vocab)
            total_ans_acc += ans_acc

            if batch_idx % self.report_freq == 0:
                self.log(
                    "| TRAIN SET | STAGE2 | "
                    + f"EPOCH [{self.current_epoch+1:02d}/{self.epochs:02d}] "
                    + f"Step [{batch_idx:04d}/{num_batches:04d}] "
                    + f"Loss: {loss.item():.4f} Ans-acc: {ans_acc:.4f}"
                )

        # calculate epoch stats
        avg_loss = total_loss / num_batches
        avg_ans_acc = total_ans_acc / num_batches
        self.train_loss.append(avg_loss)
        self.train_ans_acc.append(avg_ans_acc)

        self.log(
            f"| TRAIN_SET | EPOCH [{self.current_epoch+1:02d}/"
            + f"{self.epochs:02d}] Loss: {avg_loss:.4f} "
            + f"Ans-acc: {avg_ans_acc:.4f} "
        )
        self.evaluate_gen_qa(batch_sample)

    def val(self):
        self.vqa_model.eval()
        total_loss = 0
        total_ans_acc = 0
        total_b4 = 0
        num_batches = len(self.data_loader["valid"])

        with torch.no_grad():
            for batch_idx, batch_sample in enumerate(self.data_loader["valid"]):
                # get validation data
                image = batch_sample["image"].to(config.DEVICE)
                qa_str = batch_sample["qa_str"].to(config.DEVICE)
                image_name = batch_sample["image_name"]

                # calculate loss
                qa_out = self.vqa_model(image, qa_str)
                qa_flat = qa_str[:, 1:].flatten()
                qa_pred_flat = qa_out[:, :-1].flatten(end_dim=1)
                loss = self.criterion(qa_pred_flat, qa_flat)
                total_loss += loss.item()

                # calculate accuracy
                qa_pred = torch.argmax(qa_out, dim=2)
                ans_acc = unified_ans_acc(qa_str, qa_pred, self.unified_vocab)
                total_ans_acc += ans_acc

                # calculate bleu score
                qa_gen = self.vqa_model.generate(image)
                b4 = calc_bleu_scores_unified(
                    image_name, qa_gen, self.unified_vocab, self.vqa_struct
                )
                total_b4 += b4

                if batch_idx % self.report_freq == 0:
                    self.log(
                        "| VAL SET | "
                        + f"EPOCH [{self.current_epoch+1:02d}/{self.epochs:02d}] "
                        + f"Step [{batch_idx:04d}/{num_batches:04d}] "
                        + f"Loss: {loss.item():.4f} Ans-acc: {ans_acc:.4f} "
                        + f"BLEU4: {b4:.4f} "
                    )

        # print stats
        avg_loss = total_loss / num_batches
        avg_ans_acc = total_ans_acc / num_batches
        avg_b4 = total_b4 / num_batches
        self.val_loss.append(avg_loss)
        self.val_ans_acc.append(avg_ans_acc)
        self.val_b4.append(avg_b4)

        self.log(
            f"| VAL_SET | EPOCH [{self.current_epoch+1:02d}/"
            + f"{self.epochs:02d}] Loss: {avg_loss:.4f} "
            + f"Ans-acc: {avg_ans_acc:.4f} "
            + f"BLEU4: {avg_b4:.4f} "
        )
