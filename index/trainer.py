import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir,set_color,get_local_time,delete_file
import os

import heapq
class Trainer(object):

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")


    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, data in enumerate(iter_data):
            if isinstance(data, dict):
                embedding = data['embedding'].to(self.device)
                labels = data['labels']
            else:
                embedding = data.to(self.device)
                labels = torch.empty(0)
            self.optimizer.zero_grad()
            out, rq_loss, indices = self.model(embedding)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=embedding)
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            # print(self.scheduler.get_last_lr())
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()

        return total_loss, total_recon_loss

    #将索引转为正确的码本编号，打印出来看看
        #    比如以下结果表明有4类样本：
        #         第0类样本和其他样本区别度较大，84%都识别正确了，且映射到码本的第一个码
        #         第1类样本有一大半映射到了第4个码本
        #         但是第2和3类样本没有区分开来，都映射到了第3个码，最好是分别映射到第2、3个码，不要重。
        #       [(0, 8430), (1, 623), (3, 593), (2, 354)]
        #       [(3, 7096), (2, 2000), (1, 851), (0, 53)]
        #       [(3, 5240), (1, 2560), (2, 1791), (0, 409)]
        #        ****************Error************
        #       [(1, 5523), (2, 3691), (3, 693), (0, 93)]
    def convert_true_lable_to_cbindex(self, indices, labels, level=0):
        #首先获得labels当前这一级别(level级别)的不同编号，映射到0,1,2,...
        mapping={}
        label = labels[:, level]
        if isinstance(label, np.ndarray):
            sorted_label = np.unique(label)
            size = len(sorted_label)
            for idx, v in enumerate(sorted_label):
                mapping[idx] = v
        else:
            label = label.cpu()
            sorted_tensor, idx = torch.unique(label).sort()
            size = idx.numpy()[-1]+1
            for i in idx.numpy():
                mapping[i] = sorted_tensor[i].item()

        #S2. 分别打印每一个类别通过码本获得的索引编号，比如看看beauty这个一类别在码本中分别被映射到哪个码本，每个码本映射的数量是多少
        #    最佳的结果是所有beauty的样本的码本索引都一样，toys类别的样本的码本indices都一样（但和beauty是不一样的），这才是最佳结果

        new_label = np.zeros(label.shape[0], dtype=np.int64)
        used_codes = set()
        if level == 0:
            print("codebook-l0-result:")
        for i in range(size):
            index = np.where(label == mapping[i])[0]
            unique_elements, counts = np.unique(indices[:,level][index], return_counts=True)
            sorted_counts = sorted(zip(unique_elements, counts), key=lambda x: x[1], reverse=True)
            print(sorted_counts)
            # 用以下方法可以打印某些二级分类情况
            # print("" if level == 0 else "\t", sorted_counts)
            # if level == 0 and (i == 2 or i == 3):
            #     sub_index = index[np.where(indices[:,level][index] == i)] #在这个一级码本 i 下正确的分类的样本下标，看看他们的二级分类效果如何
            #     self.convert_true_lable_to_cbindex(indices[sub_index], labels[sub_index], level+1)
            if sorted_counts[0][0] not in used_codes:
                used_codes.add(sorted_counts[0][0])
            else:
                #发现两类应该区别开来的类别居然都大部分映射到同一个码去了
                print("****************Error************")
            new_label[index]=sorted_counts[0][0]
        return new_label

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )

        indices_set = set()
        num_sample = 0
        #hwt 修改
        all_indices = []
        all_labels = []
        for batch_idx, data in enumerate(iter_data):
            if isinstance(data, dict):
                embedding = data['embedding'].to(self.device)
                labels = data['labels']
            else:
                embedding = data.to(self.device)
                labels = torch.empty(0)
            num_sample += len(embedding)
            indices = self.model.get_indices(embedding)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

            for v in indices:
                all_indices.append(v)
            for v in labels.numpy():
                all_labels.append(v)

        all_indices = np.vstack(all_indices)
        all_labels = np.vstack(all_labels)
        _ = self.convert_true_lable_to_cbindex(all_indices, all_labels, level=0)

        collision_rate = (num_sample - len(list(indices_set)))/num_sample

        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"


    def fit(self, data):
        #第一步：初始化模型参数
        if self.args.init_method in ["load_from_ckpt"]: #从init_state文件里初始化
            state = torch.load(f"{self.args.ckpt_dir}/init_state")
            self.model.load_state_dict(state['state_dict'])
        elif self.args.init_method in ["full_init"]:
            self.model.eval()
            self.model.vq_initialization(data.dataset['labels'], data.dataset['embedding'].to(self.device))
            self._save_checkpoint(epoch=0, ckpt_file="init_state")
        else:
            print("**************** You donot init codebook first!!! ****************")

        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            if epoch_idx == 0:
                _ = self._valid_epoch(data)

            # train
            training_start_time = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            self.logger.info(train_loss_output)


            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1


                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])



        return self.best_loss, self.best_collision_rate




