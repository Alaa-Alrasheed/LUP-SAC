# =============================================================================
# federated_main_sac.py — FL main loop with SAC agent + fixed byz tracking.
#
# SAC SPECIFIC UPDATES:
#   1. Byz_Bypass_Rate: computed as actual count of byzantine indices in
#      the selected set, NOT the hardcoded 0 from aggregate().
#   2. Reward: Composite R_dir + R_mag + R_var (anti-hijacking hardened).
#      R_dir=cosine similarity, R_mag=magnitude explosion penalty,
#      R_var=internal cluster variance penalty.
#   3. SAC continuous α ∈ [0,1].
#
# OUTPUT: 3 unified CSV files (all attacks in each, with Attack column)
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from data_loader import get_dataset
from running import benignWorker, byzantineWorker
from models import CNN, ResNet18, CifarCNN, RNNClassifier, MLP
from aggregators import aggregator
from attacks import attack
from options import args_parser
import tools
import time
import copy
import warnings
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from sac_agent import SACAgent, CompositeRewardCalculator

torch.manual_seed(0)
np.random.seed(0)
warnings.filterwarnings("ignore")


# ========================= Evaluation ========================= #

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate global model. Returns dict of metrics."""
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    all_labels, all_preds = [], []
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    return {
        'accuracy': 100.0 * correct / total,
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'test_loss': total_loss / max(len(test_loader), 1),
    }


# ========================= CSV Loggers ========================= #

class CSVLogger:
    def __init__(self, filepath, columns):
        self.filepath = filepath
        self.columns = columns
        self.rows = []

    def write_header(self):
        with open(self.filepath, 'w') as f:
            f.write(','.join(self.columns) + '\n')

    def log(self, row_dict):
        self.rows.append(row_dict)
        with open(self.filepath, 'a') as f:
            f.write(','.join(str(row_dict.get(c, '')) for c in self.columns) + '\n')


def create_loggers(dataset, gar_name, skew):
    suffix = f"{dataset}_{gar_name}_{skew}"

    all_cols = [
        'Attack', 'Epoch',
        'Train_Loss', 'Test_Accuracy', 'Precision', 'Recall', 'F1_Score', 'Test_Loss',
        'Byz_Bypass_Count', 'Byz_Bypass_Rate', 'Benign_Selection_Rate', 'Num_Selected',
        'Alpha', 'Critic_Loss', 'Actor_Loss', 'Entropy_Coeff',
        'Reward', 'Cosine_Sim', 'R_dir', 'R_mag', 'R_rep',
        'Aggregation_Time_s',
    ]
    all_logger = CSVLogger(f"all_results_{suffix}.csv", all_cols)
    all_logger.write_header()

    paper_cols = [
        'Attack', 'Epoch',
        'Test_Accuracy', 'Precision', 'Recall', 'F1_Score', 'Test_Loss',
        'Byz_Bypass_Count', 'Byz_Bypass_Rate', 'Benign_Selection_Rate', 'Num_Selected',
        'Aggregation_Time_s',
    ]
    paper_logger = CSVLogger(f"paper_results_{suffix}.csv", paper_cols)
    paper_logger.write_header()

    sac_cols = [
        'Attack', 'Epoch',
        'Alpha', 'Critic_Loss', 'Actor_Loss', 'Entropy_Coeff',
        'Reward', 'Cosine_Sim', 'R_dir', 'R_mag', 'R_rep',
        'Replay_Buffer_Size',
    ]
    sac_logger = CSVLogger(f"sac_results_{suffix}.csv", sac_cols)
    sac_logger.write_header()

    return all_logger, paper_logger, sac_logger


def print_attack_summary(attack_name, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    for col in ['Test_Accuracy', 'Train_Loss', 'Critic_Loss',
                'Reward', 'Benign_Selection_Rate', 'Byz_Bypass_Rate']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f'\n{"─"*60}')
    print(f'  Summary for attack: {attack_name}')
    print(f'{"─"*60}')
    print(f'  Final Accuracy:     {df["Test_Accuracy"].iloc[-1]:.2f}%')
    print(f'  Best  Accuracy:     {df["Test_Accuracy"].max():.2f}%  '
          f'(epoch {df["Test_Accuracy"].idxmax() + 1})')
    print(f'  Avg Byz Bypass:     {df["Byz_Bypass_Rate"].mean():.4f}')
    print(f'  Avg Reward:         {df["Reward"].mean():.6f}')
    print(f'  Avg Critic Loss:    {df["Critic_Loss"].mean():.6f}')
    print(f'{"─"*60}')


# ========================= Training Function ========================= #

def train_one_round(args, train_loader, global_model, epoch,
                    Attack, GAR, sac_agent, reward_calc,
                    previous_iteration_grads_epoch,
                    score_matrix_client,
                    prev_global_grad_np):
    """
    One round of federated training with SAC.

    Returns
    -------
    (iter_loss, previous_iteration_grads_it, running_time,
     num_benign, selected_idx, alpha_val,
     sac_losses, reward, cos_sim,
     new_global_grad_np)
    """
    num_users = args.num_users
    num_byzs = args.num_byzs
    device = args.device

    # ── Local training ──
    model_copies = []
    opt = []
    for idx in range(num_users):
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(global_model.state_dict())
        model_copies.append(local_model)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr)
        opt.append(optimizer)

    m = max(int(args.frac * num_users), 1)
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_losses = []
    benign_grads = []
    byz_grads = []
    local_models = []
    user_grad_org_all = []

    for idx in idx_users[:num_byzs]:
        grad, loss, user_grad_org, model = byzantineWorker(
            model_copies[idx], opt[idx], train_loader[idx], args, idx)
        byz_grads.append(grad.clone().detach())
        user_grad_org_all.append(user_grad_org)
        local_models.append(model)

    for idx in idx_users[num_byzs:]:
        grad, loss, user_grad_org, model = benignWorker(
            model_copies[idx], opt[idx], train_loader[idx], device, args, idx)
        benign_grads.append(grad.clone().detach())
        local_losses.append(loss)
        user_grad_org_all.append(user_grad_org)
        local_models.append(model)

    user_grad_org_test = torch.zeros_like(benign_grads[0])

    # ── Apply attack ──
    byz_grads = Attack(byz_grads, benign_grads, GAR)
    local_grads = benign_grads + byz_grads
    num_benign = len(benign_grads)

    previous_grad = (user_grad_org_test if epoch == 0
                     else previous_iteration_grads_epoch)

    # ══════════════════════════════════════════════════════════════
    #  SIMPLIFIED REWARD: complete pending transition from previous
    #  round using IMMEDIATE cosine similarity (no delay).
    # ══════════════════════════════════════════════════════════════
    reward = 0.0
    cos_sim = 0.0
    r_dir = 0.0
    r_mag = 0.0
    r_rep = 0.0

    if sac_agent.has_pending() and epoch > 0 and prev_global_grad_np is not None:
        reward, cos_sim, r_dir, r_mag, r_rep = reward_calc.compute_reward(prev_global_grad_np)

        # Build next_state (a rough approximation — the real state
        # comes from the aggregation below, but we need something
        # to close the previous transition)
        dummy_next = np.zeros(12, dtype=np.float32)
        done = (epoch >= args.epochs - 1)
        sac_agent.complete_pending(reward, dummy_next, done)

    # ══════════════════════════════════════════════════════════════
    #  AGGREGATION — LUP_SAC with continuous weighting
    # ══════════════════════════════════════════════════════════════
    start_time = time.time()
    global_grad, selected_idx, alpha_val = GAR.aggregate(
        local_grads, user_grad_org_all, previous_grad,
        score_matrix_client,
        f=num_byzs, epoch=epoch, g0='grad_0', iteration=0,
        sac_agent=sac_agent, reward_calc=reward_calc,
    )
    end_time = time.time()
    running_time = end_time - start_time

    # ── SAC training step ──
    sac_losses = sac_agent.update_model()

    # ── Update reward calculator ──
    new_global_grad_np = global_grad.detach().cpu().numpy().flatten()
    reward_calc.update_ema_grad(new_global_grad_np)

    loss_avg = sum(local_losses) / len(local_losses) if local_losses else 0.0
    reward_calc.update_loss_history(loss_avg)

    # ── Update global model ──
    global_model.load_state_dict(local_models[0].state_dict())
    tools.set_gradient_values(global_model, global_grad)
    previous_iteration_grads_it, _ = tools.get_gradient_values(global_model)

    print(
        f'[epoch {epoch+1}, {100*(epoch+1)/args.epochs:.2f}%] '
        f'loss: {loss_avg:.5f} --- '
        f'\u03b1: {alpha_val:.4f} --- '
        f'critic_loss: {sac_losses["critic_loss"]:.6f} --- '
        f'entropy_coeff: {sac_losses["alpha"]:.4f}')

    return ([loss_avg], previous_iteration_grads_it, running_time,
            num_benign, selected_idx, alpha_val,
            sac_losses, reward, cos_sim, r_dir, r_mag, r_rep,
            new_global_grad_np)


# =============================== MAIN =============================== #

if __name__ == '__main__':
    attack_keys = [
        'random', 'noise', 'label_flip', 'byzMean', 'sign_flip',
        'lie', 'min_max', 'min_sum', 'mpfa'
    ]

    GAR_keys = ['LUP_SAC']

    for gar_name in GAR_keys:
        args = args_parser()
        args.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        device = args.device

        train_loader, test_loader = get_dataset(args)

        all_logger, paper_logger, sac_logger = create_loggers(
            args.dataset, gar_name, args.skew)

        print(f"\n{'='*60}")
        print(f"  Output files:")
        print(f"    {all_logger.filepath}")
        print(f"    {paper_logger.filepath}")
        print(f"    {sac_logger.filepath}")
        print(f"{'='*60}")

        # ════════════════════════════════════════════
        #  LOOP OVER ATTACKS
        # ════════════════════════════════════════════
        attacks_to_run = args.attack if args.attack and len(args.attack) > 0 else attack_keys
        for attack_name in attacks_to_run:
            print(f"\n{'='*60}")
            print(f"  Dataset={args.dataset} | Attack={attack_name} | "
                  f"Defense={gar_name}")
            print(f"  Clients={args.num_users} | Byzantines={args.num_byzs} | "
                  f"Epochs={args.epochs}")
            print(f"  Device: {device}")
            print(f"{'='*60}")

            # Fresh model per attack
            if args.dataset == 'ton_iot':
                global_model = MLP().to(device)
            elif args.dataset == 'AGNews':
                global_model = RNNClassifier().to(device)
            elif args.dataset == 'cifar':
                global_model = CifarCNN().to(device)
            elif args.dataset == 'fmnist':
                global_model = CNN().to(device)
            else:
                global_model = CNN().to(device)

            criterion = nn.CrossEntropyLoss()
            Attack_fn = attack(attack_name)
            GAR = aggregator(gar_name)()

            # Fresh SAC agent per attack
            sac_agent = SACAgent(
                state_dim=12, hidden_dim=64,
                lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                gamma=0.99, tau=0.005,
                buffer_size=10_000, batch_size=64,
                init_alpha=0.2, target_entropy=-0.5,
                device=str(device),
            )
            reward_calc = CompositeRewardCalculator(
                ema_alpha=0.1,
                mag_threshold_factor=2.0,
                mag_penalty=-1.0,
            )

            # Book-keeping
            previous_iteration_grads_epoch = []
            score_matrix_client = np.zeros([args.num_users, 1])
            prev_global_grad_np = None
            attack_rows = []

            # ════════════════════════════════════════
            #  EPOCH LOOP
            # ════════════════════════════════════════
            for epoch in range(args.epochs):
                (loss, previous_iteration_grads_epoch, running_time,
                 num_benign, selected_idx, alpha_val,
                 sac_losses, reward, cos_sim, r_dir, r_mag, r_rep,
                 prev_global_grad_np) = train_one_round(
                    args, train_loader, global_model, epoch,
                    Attack_fn, GAR, sac_agent, reward_calc,
                    previous_iteration_grads_epoch,
                    score_matrix_client,
                    prev_global_grad_np)

                # ══════════════════════════════════════════
                #  FIX: Correct Byzantine bypass calculation
                # ══════════════════════════════════════════
                # In local_grads: indices 0..num_benign-1 are benign,
                #                 indices num_benign..N-1 are byzantine.
                # Any selected index >= num_benign is a byz that bypassed.
                byz_bypassed = sum(1 for idx in selected_idx
                                   if idx >= num_benign)
                byz_bypass_rate = byz_bypassed / max(args.num_byzs, 1)

                benign_in_selected = sum(1 for idx in selected_idx
                                         if idx < num_benign)
                benign_sel_rate = benign_in_selected / max(num_benign, 1)

                metrics = evaluate_model(
                    global_model, test_loader, criterion, device)
                print(f"Test Accuracy: {metrics['accuracy']:.2f}%  |  "
                      f"Byz Bypassed: {byz_bypassed}/{args.num_byzs}  |  "
                      f"Benign Kept: {benign_in_selected}/{num_benign}")

                train_loss_val = loss[0] if loss else 0.0

                row = {
                    'Attack': attack_name,
                    'Epoch': epoch + 1,
                    'Train_Loss': f'{train_loss_val:.6f}',
                    'Test_Accuracy': f'{metrics["accuracy"]:.2f}',
                    'Precision': f'{metrics["precision"]:.4f}',
                    'Recall': f'{metrics["recall"]:.4f}',
                    'F1_Score': f'{metrics["f1"]:.4f}',
                    'Test_Loss': f'{metrics["test_loss"]:.6f}',
                    'Byz_Bypass_Count': byz_bypassed,
                    'Byz_Bypass_Rate': f'{byz_bypass_rate:.4f}',
                    'Benign_Selection_Rate': f'{benign_sel_rate:.4f}',
                    'Num_Selected': len(selected_idx),
                    'Alpha': f'{alpha_val:.4f}',
                    'Critic_Loss': f'{sac_losses["critic_loss"]:.6f}',
                    'Actor_Loss': f'{sac_losses["actor_loss"]:.6f}',
                    'Entropy_Coeff': f'{sac_losses["alpha"]:.4f}',
                    'Reward': f'{reward:.6f}',
                    'Cosine_Sim': f'{cos_sim:.6f}',
                    'R_dir': f'{r_dir:.6f}',
                    'R_mag': f'{r_mag:.6f}',
                    'R_rep': f'{r_rep:.6f}',
                    'Aggregation_Time_s': f'{running_time:.4f}',
                    'Replay_Buffer_Size': len(sac_agent.memory),
                }

                all_logger.log(row)
                paper_logger.log(row)
                sac_logger.log(row)
                attack_rows.append(row)

            print_attack_summary(attack_name, attack_rows)

            ckpt_path = (f"sac_ckpt_{args.dataset}_{attack_name}"
                         f"_{args.num_byzs}_{gar_name}.pt")
            sac_agent.save(ckpt_path)
            print(f"  [\u2713] SAC checkpoint saved to {ckpt_path}")

        print(f"\n{'='*60}")
        print(f"  ALL EXPERIMENTS COMPLETE")
        print(f"{'='*60}")
        print(f"  All metrics:   {all_logger.filepath}")
        print(f"  Paper metrics: {paper_logger.filepath}")
        print(f"  SAC metrics:   {sac_logger.filepath}")
        print(f"  Total rows:    {len(all_logger.rows)}")
        print(f"{'='*60}\n")
