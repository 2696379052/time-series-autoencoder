import os

import torch
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from tqdm import tqdm

from .eval import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_iter, test_iter, model, criterion, optimizer, config, ts):
    """
    Training function.

    Args:
        train_iter: (DataLoader): train data iterator
        test_iter: (DataLoader): test data iterator
        model: model
        criterion: loss to use
        optimizer: optimizer to use
        config:
    """
    tb_writer_train = SummaryWriter(logdir=config.general.output_dir, filename_suffix="train")
    tb_writer_test = SummaryWriter(logdir=config.general.output_dir, filename_suffix="test")

    if not os.path.exists(config.general.output_dir):
        os.makedirs(config.general.output_dir)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.lrs_step_size, gamma=0.5)

    global_step, logging_loss = 0, 0.0
    train_loss = 0.0

    # 追踪最优评估损失，用于保存最佳模型
    best_eval_loss = float("inf")
    best_epoch = -1
    best_ckpt_path = os.path.join(config.general.output_dir, "checkpoint-best.ckpt")

    # 预计算正则化参数列表（避免每 batch 重复筛选）
    reg_params = None
    if config.training.reg1 or config.training.reg2:
        reg_params = [p for name, p in model.named_parameters() if "bias" not in name]

    # 混合精度训练设置
    use_amp = getattr(config.training, "use_amp", False) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("✓ 启用混合精度训练 (AMP)")

    for epoch in tqdm(range(config.training.num_epochs), unit="epoch"):
        model.train()  # 每个 epoch 开始时设置一次
        
        for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), unit="batch"):
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

            feature, y_hist, target = batch
            feature = feature.to(device)
            y_hist = y_hist.to(device)
            target = target.to(device)

            # 混合精度前向传播
            with autocast(enabled=use_amp):
                output = model(feature, y_hist)
                loss = criterion(output, target)

                # 使用预计算的参数列表计算正则化
                if config.training.reg1 and reg_params:
                    params_flat = torch.cat([p.view(-1) for p in reg_params])
                    loss = loss + config.training.reg_factor1 * torch.norm(params_flat, 1)
                if config.training.reg2 and reg_params:
                    params_flat = torch.cat([p.view(-1) for p in reg_params])
                    loss = loss + config.training.reg_factor2 * torch.norm(params_flat, 2)

                if config.training.gradient_accumulation_steps > 1:
                    loss = loss / config.training.gradient_accumulation_steps

            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            train_loss += loss.item()

            if (i + 1) % config.training.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

                if global_step % config.general.logging_steps == 0:
                    # 仅记录训练损失和学习率，不在这里做评估
                    tb_writer_train.add_scalar(
                        "train_loss",
                        (train_loss - logging_loss) / config.general.logging_steps,
                        global_step,
                    )
                    tb_writer_train.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    logging_loss = train_loss

            if global_step % config.general.save_steps == 0:
                torch.save({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "decoder_state_dict": model.decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_fn": criterion,
                }, os.path.join(config.general.output_dir, f"checkpoint-{global_step}.ckpt"))

        # 每个 epoch 结束后做一次评估，并根据 loss 保存最佳模型
        if config.general.do_eval:
            print("\n================ 评估开始 (epoch = {}, global_step = {}) ================".format(
                epoch + 1, global_step
            ))
            results = evaluate(test_iter, criterion, model, config, ts)

            # 写入 TensorBoard
            for key, val in results.items():
                tb_writer_test.add_scalar(f"eval_{key}", val, global_step)

            # 在控制台展示评估结果
            for key, val in results.items():
                try:
                    print(f"  {key}: {float(val):.6f}")
                except (TypeError, ValueError):
                    print(f"  {key}: {val}")

            current_loss = results.get("loss")
            if current_loss is not None and current_loss < best_eval_loss:
                best_eval_loss = current_loss
                best_epoch = epoch + 1
                torch.save({
                    "epoch": best_epoch,
                    "global_step": global_step,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "decoder_state_dict": model.decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_results": results,
                }, best_ckpt_path)
                print(
                    "✓ 发现更优模型: epoch = {}, loss = {:.6f}，已保存到 {}".format(
                        best_epoch, best_eval_loss, best_ckpt_path
                    )
                )
            else:
                print(
                    "当前 epoch 未优于历史最佳: 当前 loss = {}, 最佳 loss = {:.6f} (epoch = {})".format(
                        results.get("loss"), best_eval_loss, best_epoch
                    )
                )
