# Clean Sweep 最終結果與診斷

**日期：** 2026-05-12
**範圍：** Speaker × Aug 4-cell ablation（A/B/C/D），de-leaked 重跑後在 50-clip VB eval 上的最終結果，以及 hallucination tail 的成因診斷。

---

## 0. 背景

- 原本的 A/B/C/D（以及 E/F/G/H 擴展）有兩層 leakage：
  - **聲學**：XTTS speaker references 取自 VB eval 的 source clips
  - **文本**：598 entry 訓練 corpus 中有 98 entry 是 VB human transcripts 原樣 copy
- 2026-05-11 啟動 clean rerun：
  - Speaker refs 改用 Broadcastify Boston EMS feed 36636（與 VB 的 14744 feed 完全不同）→ [broadcastify_seeds/speaker_profiles.json](broadcastify_seeds/speaker_profiles.json)
  - Corpus 過濾掉 `source=human`，剩下 500 entry → `combined_corpus_llm_only.jsonl`
- 2026-05-12 凌晨跑完，4 個 LoRA 模型 + aligned inference CSV 全部產出於 [speaker_aug_sweep_clean/](speaker_aug_sweep_clean/)

舊 leaky 報告詳見已存在的 sweep 分析文件；本文件只記錄 clean rerun 之後的事實。

---

## 1. Clean rerun 四個 condition 的 aggregate WER

評估方式：clip-level WER，50 個 VB clip，paired bootstrap 10k iters（[bootstrap_wer_ci.py](bootstrap_wer_ci.py)）。

| Condition       | WER     | 95% CI            | Δ vs base   | 95% Δ-CI          | >100% WER | worst clip |
|-----------------|---------|-------------------|-------------|-------------------|-----------|------------|
| base (no FT)    | 54.52%  | [49.54, 59.81]    | —           | —                 | 1/50      | 160.6%     |
| A (5spk, orig)  | 55.26%  | [49.42, 61.82]    | +0.74pp     | [−2.77, +4.64]    | 3/50      | 159.0%     |
| B (20spk, orig) | 54.67%  | [49.05, 60.82]    | +0.15pp     | [−3.24, +3.63]    | 3/50      | 148.5%     |
| C (5spk, enh)   | 55.64%  | [49.77, 62.18]    | +1.12pp     | [−2.38, +4.87]    | 3/50      | 149.2%     |
| D (20spk, enh)  | 57.15%  | [50.92, 64.05]    | +2.63pp     | [−0.90, +6.48]    | 2/50      | **221.2%** |

重跑指令：`python3 analyze_clean_sweep_tail.py`

### 1.1 重點觀察

1. **沒有任何 FT condition 統計上贏 base**。所有 Δ-CI 跨 0。原本 leaky sweep 給的「B −1.94pp」是 leakage artifact，clean 之後 B 變成 +0.15pp（甚至比 base 略差）。
2. **FT 普遍放大 hallucination tail**。base 只有 1/50 clip WER > 100%，A/B/C 都是 3/50（×3 倍），D 雖然只有 2/50 但 worst clip 拉到 221.2%。
3. **「speaker 多 + 強 aug」並沒有加成**。D（20spk + enhanced aug）是 4 個裡最差的。Enhanced aug 在 5spk 跟 20spk 都讓結果更糟，從來沒幫上忙。
4. **真實 FT 效果是 indistinguishable from zero**。50 clip 的 noise floor 約 ±3pp，目前所有觀察值都在這個範圍內。

---

## 2. Guarded decoding probe（B only）

**問題：** B 的 +60pp tail 是 fine-tuning bias，還是 decoding-time issue（greedy 衝太快、temp 沒 fallback）？

**做法：** 對 base 跟 B 都跑一次 guarded decoding（temperature fallback 0→0.2→0.4→0.6→0.8 + `compression_ratio_threshold=2.4` + `logprob_threshold=−1.0`），與 greedy 對比。

腳本：[run_whisper_inference_guarded.py](run_whisper_inference_guarded.py)；分析：[analyze_guarded_b.py](analyze_guarded_b.py)

### 2.1 Aggregate 結果

| Condition       | WER     | Δ vs greedy base | 95% Δ-CI          | P(< base) |
|-----------------|---------|------------------|-------------------|-----------|
| base (greedy)   | 54.52%  | —                | —                 | —         |
| base_guarded    | 54.64%  | +0.12pp          | [−0.07, +0.48]    | 0.233     |
| B (greedy)      | 54.67%  | +0.15pp          | [−3.24, +3.63]    | 0.470     |
| B_guarded       | 54.99%  | +0.47pp          | [−2.81, +3.83]    | 0.386     |

50 個 clip 的 per-clip Δ（guarded − unguarded）：

| Metric              | mean   | median | p10  | p90  | min    | max    |
|---------------------|--------|--------|------|------|--------|--------|
| Δ(base → base_grd)  | +0.35  | +0.00  | 0.00 | 0.00 | −0.62  | +18.18 |
| Δ(B → B_grd)        | +0.73  | +0.00  | 0.00 | 0.00 | −9.43  | +39.39 |

### 2.2 Tail 沒被救回

B 最爛的 5 個 clip，guard 前後對照：

| Clip                                            | base  | B     | B_grd  | base_grd |
|-------------------------------------------------|-------|-------|--------|----------|
| 202412011757-965821-14744_call_11.wav           | 72.5% | 145.0%| 145.0% | 72.5%    |
| 202412041604-460429-14744_call_10.wav           | 82.0% | 124.6%| 119.7% | 82.0%    |
| 202412040736-239835-14744_call_4.wav            | 54.9% | 81.7% | 81.7%  | 54.9%    |
| 202412050858-692837-14744_call_8.wav            | 62.9% | 85.5% | 85.5%  | 62.9%    |
| 202412021121-31317-14744_call_2.wav             | 50.6% | 64.7% | 64.7%  | 50.6%    |

Tail-clip 數：

| Condition       | >100% WER | worst clip |
|-----------------|-----------|------------|
| base            | 1/50      | 160.6%     |
| base_guarded    | 1/50      | 178.8%     |
| B               | 3/50      | 148.5%     |
| B_guarded       | 3/50      | 187.9%     |

### 2.3 結論

- Guarded decoding 在這個 dataset 上 median Δ = 0、CI 緊貼 0，aggregate 上幾乎是 no-op。
- B 的 hallucination tail **完全沒被 guarding 砍掉**，最壞值反而略增（temp fallback 偶爾會生出更長的 retry hypothesis）。
- 因此 B 的 +60pp tail 不是 decoding artifact，是 fine-tuning 把 token 分佈推到 over-generation 的訓練分佈效應。
- A/C/D 不需要補 guarded inference — guarding 對 base+B 都是 no-op，沒有理由相信 A/C/D 會不同。

---

## 3. B 的 3 個 tail clip 實際吐了什麼

**問題：** FT 到底把 B 推往哪個方向，導致 +60pp tail？

腳本：[analyze_b_tail_clips.py](analyze_b_tail_clips.py)

### 3.1 Token 數對照

| Clip            | ref tok | base tok | **B tok** | base WER | B WER  | Δ        |
|-----------------|---------|----------|-----------|----------|--------|----------|
| call_3          | 33      | 63       | 61        | 160.6%   | 148.5% | −12.1pp  |
| call_11         | 40      | 62       | **90**    | 72.5%    | 145.0% | +72.5pp  |
| call_10         | 61      | 77       | **106**   | 82.0%    | 124.6% | +42.6pp  |

call_3 不是 B 引起的 — base 自己已經幻覺到 160%（吐 "Ministry of Local Office"、"Mark Rowe"），B 反而少幾個字。**真正的 FT-induced 災難在 call_11 與 call_10。**

### 3.2 call_11（B 多幻覺出 ~28 token）

**REF（消防無線電 ladder 2 status check）：**
> command to ladder 2 status of copy comments stated theres possibly a occupant on oxygen interior 19 20 possibly one more individual in over by where were parked were making patient contact just stand by for a second will do

**BASE hyp：** 跟著 ref 內容走，只是冗長。

**B hyp（粗體為憑空多出的內容）：**
> Commander Ladder 2, what's the status of primaries? **"I'm a period primary person with a heart attack."** Copy. The comment stated there is possibly an occupant on oxygen interior. **41920. This is the fire department for our new detention center inside the room we're going to start pulling some sealants** and we understand there's still possibly one more individual in this area...

### 3.3 call_10（B 多幻覺出 ~29 token）

**REF（engine 18 → battalion 3）：**
> engine 18 to battalion 3 go ahead 18 on scene were at a shop well go ahead and pass command to you well be out investigating at dollar general 18 is on scene but nothing is showing well be out investigating ...

**BASE hyp：** 開頭直接 "Engine 18 of Italian 3..."

**B hyp（粗體為憑空多出的內容）：**
> **"This is the US Army, this is the EAS recent tail of a chief tennis player on the field. 10-4 to 1723. He's on fire 4."** Engine 18 of Italian 3. On scene, we got a uh script shot...

### 3.4 失敗模式診斷

三個 clip 一致顯示 B 學到的是「**這聲音聽起來像無線電 → 我應該輸出 radio template**」：

1. **句首 EAS-style 開場白**：「This is the US Army, this is the EAS...」 — Whisper 經典「廣播 intro 幻覺」，FT 後變嚴重。
2. **編造 callsign / 數字**：「41920」、「10-4 to 1723」、「He's on fire 4」、「Squad 15」 — 訓練 corpus 裡有大量 callsign，模型學會「不確定就填一個」。
3. **灌 EMS 關鍵字**：「heart attack」、「fire department」、「detention center」、「sealants」 — 訓練 label 都是完整 EMS 對話，模型沒看過「沉默」或「非 EMS 內容」當 label，因此用 EMS 詞彙補白。

這正是 synthetic-radio FT 的預期 failure mode：模型把「ambiguous audio → 必定是 EMS radio」這個 prior 學得太死。Greedy 跟 temperature fallback 都救不了，因為這是 **high-confidence over-generation**，不是 low-confidence retry 能砍掉的。

---

## 4. 降 epoch 重訓 B 的結果（mitigation §6.1 的驗證）

**問題：** 5 epoch on 500-entry corpus 是否就是 tail explosion 的元兇？把 epoch 砍到 1 看看 aggregate 跟 tail 怎麼變。

**做法：**
- Variant `B_clean_20spk_orig_1ep`：保留 LR 3e-5、batch 4、相同 augmented data，只把 epoch 從 5 改成 1。
- 啟動腳本同 [run_speaker_aug_sweep_clean.sh](run_speaker_aug_sweep_clean.sh) 的 step 5，只 override `--num_train_epochs 1 --eval_steps 50 --save_steps 50`。
- 訓練 ~45 min（100 steps）+ inference ~4 min。Final `train_loss=4.88` vs B_5ep `2.30`（明顯 under-trained）。

### 4.1 Aggregate（base / B_5ep / B_1ep）

| Condition | WER     | 95% CI            | Δ vs base   | 95% Δ-CI            | >100% WER | worst clip |
|-----------|---------|-------------------|-------------|---------------------|-----------|------------|
| base      | 54.52%  | [49.54, 59.81]    | —           | —                   | 1/50      | 160.6%     |
| B_5ep     | 54.67%  | [49.05, 60.82]    | +0.15pp     | [−3.24, +3.63]      | 3/50      | 148.5%     |
| **B_1ep** | 55.29%  | [49.99, 60.89]    | +0.77pp     | **[−0.37, +2.23]**  | **2/50**  | 160.6%     |

腳本：[analyze_b_epoch_sweep.py](analyze_b_epoch_sweep.py)

### 4.2 Per-clip：1 epoch 救回的 vs 搞砸的

**5ep 災難 → 1ep 救回**：

| Clip          | B_5ep WER | B_1ep WER | Δ        | 對應 §3 失敗模式 |
|---------------|-----------|-----------|----------|------------------|
| call_10       | 124.6%    | 82.0%     | **−42.6pp** | "EAS recent tail..." 開場白消失 |
| call_11       | 145.0%    | 122.5%    | −22.5pp  | "heart attack" 灌詞減少 |
| call_4        | 81.7%     | 54.9%     | −26.8pp  | — |
| call_8        | 85.5%     | 66.1%     | −19.4pp  | — |
| call_16       | 88.1%     | 78.0%     | −10.2pp  | — |

**5ep 還行 → 1ep 變差**：

| Clip          | B_5ep WER | B_1ep WER | Δ        |
|---------------|-----------|-----------|----------|
| call_12       | 46.5%     | 76.1%     | +29.6pp  |
| call_6 (1329) | 56.9%     | 81.7%     | +24.8pp  |
| call_6 (1703) | 57.1%     | 77.1%     | +20.0pp  |
| call_13       | 53.7%     | 72.0%     | +18.3pp  |
| call_14       | 47.8%     | 65.2%     | +17.4pp  |

### 4.3 結論

1. **Tail 確實是 epoch 過多的副作用**。1 epoch 把最爛的 `call_10` 從 124.6% 砍回 82.0%（接近 base 的水準），>100% WER clip 從 3 降到 2，worst clip WER 從 187%（B_guarded）/148%（B_5ep）回到 160%（≈ base）。
2. **但 1 epoch 同時也 under-trained**。原本 B_5ep 在 50-60% 區間運作正常的 clip，1 epoch 退化到 65-82%（train_loss 4.88 vs 2.30 是直接證據）。
3. **Aggregate 是 wash**：B_1ep 55.29% vs B_5ep 54.67%，Δ +0.62pp，互換失敗模式而已。但 B_1ep 的 **CI 明顯變窄** [−0.37, +2.23]（B_5ep 是 [−3.24, +3.63]） → behaviorally 更 consistent，少 outlier。
4. **失敗模式轉換**，不是消除：5 epoch = "偶爾炸裂式幻覺"，1 epoch = "整體微差但平穩"。對下游應用哪個更可接受要看 use case；如果做後處理 / human-in-the-loop，B_1ep 的可預測性可能更實用。
5. **Sweet spot 在 2-3 epoch**：尚未驗證。預期 aggregate ≈ B_5ep 或略低、tail 接近 B_1ep。下一輪可以跑 `B_2ep` 確認。

---

## 5. 合成 vs 真實 VB 的音訊層差距（為什麼 FT 沒效的物理解釋）

**問題：** 前面的結果都顯示 FT 沒幫到 VB 而且放大 tail。是不是合成音在 audio 層就跟真實 VB 不同分佈？

**做法：** 用 [ems_whisper_eval_bundle/run_gap_analysis.py](ems_whisper_eval_bundle/run_gap_analysis.py)，把真實 50 個 VB eval clip（從 random_samples_1/2 stage）跟 bundle 內三個 v6 synth 變體（v6_raw / v6_radio / v6_aug，各 50 clip）做 DSP + Whisper-encoder 距離比較。Wrapper：[analyze_vb_synth_gap.sh](analyze_vb_synth_gap.sh)，輸出在 [ems_whisper_eval_bundle/vb_gap_results/](ems_whisper_eval_bundle/vb_gap_results/)。

### 5.1 DSP 指標（mean over 50 clips）

| Metric             | **real VB** | v6_aug | v6_radio | v6_raw | 解讀 |
|--------------------|-------------|--------|----------|--------|------|
| RMS（音量）        | **0.025**   | 0.110  | 0.230    | 0.162  | 真實 VB **5-10× 較安靜** |
| **pause_pct**      | **65.6%**   | 10.0%  | 0.8%     | 2.8%   | **真實 VB 三分之二是靜音；合成幾乎沒靜音** |
| spectral centroid  | **543 Hz**  | 1563   | 1899     | 1793   | 真實 VB 頻譜重心**低 3-4 倍** |
| bandwidth          | **380 Hz**  | 743    | 1057     | 1013   | 真實 VB 頻寬只有合成的一半 |
| spectral flatness  | **0.60**    | 0.05   | 0.007    | 0.006  | 真實 VB **接近白雜訊**（重底噪），合成幾乎是純人聲 |
| ZCR                | **0.050**   | 0.184  | 0.202    | 0.188  | 真實 VB 低頻 dominant |
| f0 mean            | **148 Hz**  | 203    | 222      | 220    | 真實 VB voiced 段音高低（可能多男聲） |
| duration (clip)    | 25.0s       | 23.4s  | 23.8s    | 23.6s  | 一致，無 duration 混淆 |

### 5.2 Wasserstein + Whisper encoder 距離

Wasserstein（real vs each variant，越小越接近）：

| Feature           | v6_raw   | v6_radio | **v6_aug** |
|-------------------|----------|----------|------------|
| rms               | 0.137    | 0.205    | **0.085**  |
| centroid_hz       | 1250     | 1356     | **1020**   |
| bandwidth_hz      | 633      | 678      | **363**    |
| rolloff_hz        | 1720     | 1756     | **1387**   |
| f0_mean_hz        | 72.4     | 73.7     | **55.1**   |
| pause_pct         | 0.627    | 0.648    | **0.556**  |

Whisper-encoder（centroid_dist / MMD-RBF）：

| Pair              | centroid_dist | mean_pairwise | **MMD-RBF** |
|-------------------|---------------|---------------|-------------|
| real vs v6_raw    | 4.60          | 5.56          | 0.66        |
| real vs v6_radio  | 4.74          | 5.66          | 0.69        |
| **real vs v6_aug**| **4.08**      | **5.23**      | **0.53**    |

v6_aug 在所有指標都最接近真實 VB，**但 DSP 上仍然嚴重偏離**（RMS 4×、centroid 3×、pause 6× 差距還在）。

### 5.3 關鍵發現

1. **靜音是最大的單一差距**：真實 VB 65% pause，合成 0.8-10%。合成被當作「連續講話」訓練，遇到真實 VB 的長停頓就用 radio template 填補 — **這直接解釋了 §3.3 的 call_10「This is the US Army, this is the EAS recent tail...」幻覺**：模型從來沒在 silence 區段看過 label，所以靜音變成 radio EAS 開場白的觸發器。
2. **底噪沒被模擬**：spectral flatness 真實 0.60（接近白雜訊），合成 0.006-0.05（純人聲）。合成是「乾淨人聲 + radio EQ」，真實是「人聲 + 持續無線電底噪」。
3. **頻譜窄太多 + 重心低太多**：VB centroid 543 Hz / bandwidth 380 Hz；合成 1500-1900 Hz / 700-1000 Hz。真實手持無線電 + 老 codec 的 bandpass 比我們模擬的窄、低頻更多。
4. **音量分佈完全錯**：真實 RMS 0.025 vs 合成 0.11-0.23，差 5-10×。合成 audio 在訓練時的 loudness 分佈跟真實 VB 完全不同。
5. **Enhanced aug（v6_radio）反而最遠**：所有 DSP 指標 v6_radio 都比 v6_raw 偏離真實更多。**我們的 "radio FX" 往錯方向 over-engineer 了** — 拉高頻寬、提高 centroid，但真實 VB 是窄帶 + 低重心。這跟 §1 看到的「enhanced aug 從沒幫上忙」吻合。

### 5.4 Encoder 與 DSP 不一致的警告

Whisper encoder MMD 對 VB 的距離（0.53）反而**比對 Boston Broadcastify**（0.94，見 [ems_whisper_eval_bundle/gap_results/](ems_whisper_eval_bundle/gap_results/)）**還近**，DSP 卻差更大。

意義：encoder 在意 voice content / phonetic，會 normalize 掉 RMS / silence / 底噪。**過去用 encoder 距離當「合成像不像真實」的 proxy，會嚴重低估 domain gap**。低層統計才是真正的差距。

### 5.5 對前面結果的物理解釋

- 「**Synthetic FT 不改善 VB**」（§1）：模型學的不是 VB 的聲學特徵，是「乾淨連續人聲 + 我們想像中的 radio FX」。在 audio 分佈不一致的條件下，FT 學到的 EMS 詞彙先驗無法 transfer 到真實聲學環境。
- 「**B 的 over-generation tail**」（§3）：訓練 100% 都是「該講話的時段」，遇到 65% 靜音的真實 audio 就硬填內容 → EAS-intro 幻覺、callsign 填空、EMS 詞灌水。
- 「**Guard decoding 救不了 tail**」（§2）：因為這不是低信心 retry 能砍掉的，是訓練分佈跟測試分佈在低層就分歧。
- 「**Enhanced aug 從來沒幫上忙**」（§1 + §4）：因為它把合成推向更高頻寬、更高 centroid，**遠離**真實 VB 的窄帶低重心特性。

---

## 6. 對「synthetic FT 能不能改善 VB」這個問題的當前答案

基於 clean rerun + guarded probe + tail debug + B_1ep mitigation + VB 音訊層分析：

- **aggregate 上：FT 效果與 0 不可區分**。所有 condition（A/B/C/D/B_1ep）的 Δ-CI 都包含 0。
- **tail 上：FT 確定讓事情變糟**。Epoch 數是槓桿之一（B_1ep 把最壞 clip 拉回 base 水準）但不是唯一原因。
- **根本原因在音訊層的 domain mismatch**（§5）：合成 audio 是「乾淨連續人聲 + 想像中的 radio FX」，真實 VB 是「窄帶低重心人聲 + 65% 靜音 + 持續底噪」。模型學的是錯的聲學分佈。
- **Enhanced aug 不是「不夠強」，是方向錯**：v6_radio 在所有 DSP 指標上都比 v6_raw 偏離真實更多 — 我們在拉高頻寬，但真實 VB 是窄帶。
- **這個方向的 ceiling 比想像低很多**。原本 leaky 看到的 −1.4 ~ −1.9pp 改善幾乎全部是 leakage。Synthetic-FT 路線要繼續推，必須先解決 audio-domain mismatch，否則任何 epoch / LR / corpus 微調都是在分佈外的空間優化。

---

## 7. 後續可行方向

按工作量由小到大（標 ✓ 為已驗證、❌ 為已試但結論「不夠」）：

1. **❌ 降 epoch / 降 LR 重訓 B**（2026-05-12 跑完）：1 epoch 救回最壞 tail clip 但 mid-range 退化，aggregate 平手。**Sweet spot 還沒測**（2-3 epoch）。詳見 §4。
2. **🎯 修 audio-domain mismatch**（§5 的直接 actionable list — 工程量小、可能 ROI 最大）：
   - **加靜音**：訓練 audio 後處理插入 60-65% pause（match VB pause_pct）。應該直接砍 §3 的 EAS-intro 幻覺。
   - **降 RMS**：訓練 audio normalize 到 RMS ≈ 0.025（match VB loudness）。
   - **加底噪**：疊上窄帶白雜訊 / radio static 直到 spectral flatness ≈ 0.6。
   - **改 bandpass**：augmentation 用 500-1500Hz 窄帶 + 低頻 boost，而不是 300-3400Hz 寬頻。
3. **Corpus 加 negative examples**：silence、雜訊、非 EMS 語音段 + 空 label。與 (2) 互補 — (2) 是改 input 分佈，這是改 (input, label) 對應。
4. **加 KL penalty 防 prior drift**：訓練 objective 加 `KL(p_ft || p_base)`。理論上同時保 mid-range + 控 tail。需改 finetune 程式。
5. **Broadcastify upper-bound**（科學上最有價值但工程量最大）：用 real EMS radio audio 做 XTTS seed 重訓，量化「合成 vs 真實」的天花板。如果即使用真實 audio 當 seed 仍只能 wash，那 synthetic 路線本質有問題；如果能贏 base，那 (2)(3)(4) 才值得繼續細調。Seed 已建好（40 ref + speaker_profiles.json），下一步 XTTS 合成 → 訓練。

**新建議優先序**：
- 先 (2) 一條條測：每條 mitigation 重訓一個 B 變體，看 §5 的 DSP 指標 + VB WER 同時往哪裡走。靜音 + RMS 兩條應該最便宜（純後處理）。
- 同時跑 (5)：給 ceiling 答案。如果 ceiling 低，後面 mitigation 也不用做了。
- (1) 找 sweet spot 可以順手放在 (2) 旁邊。
- (3)(4) 等 (2)(5) 結論再決定。

---

## 8. 相關檔案速查

- 訓練輸出與 inference CSV：[speaker_aug_sweep_clean/](speaker_aug_sweep_clean/)
- Aligned eval reference：[vb_aligned_eval/](vb_aligned_eval/)（含 [base_whisper_large_v3.csv](vb_aligned_eval/base_whisper_large_v3.csv) 與 [base_whisper_large_v3_guarded.csv](vb_aligned_eval/base_whisper_large_v3_guarded.csv)）
- Human annotation：[vb_ems_anotation/human_anotation_vb.csv](vb_ems_anotation/human_anotation_vb.csv)
- Bootstrap 與正規化：[bootstrap_wer_ci.py](bootstrap_wer_ci.py)、[ems_eval/preprocessing.py](ems_eval/preprocessing.py)
- 分析腳本：
  - [analyze_clean_sweep_tail.py](analyze_clean_sweep_tail.py)（產出 §1 的表）
  - [analyze_guarded_b.py](analyze_guarded_b.py)（產出 §2 的表）
  - [analyze_b_tail_clips.py](analyze_b_tail_clips.py)（產出 §3 的 raw hyp 對照）
  - [analyze_b_epoch_sweep.py](analyze_b_epoch_sweep.py)（產出 §4 的 base/5ep/1ep 對照）
  - [analyze_vb_synth_gap.sh](analyze_vb_synth_gap.sh) → [ems_whisper_eval_bundle/run_gap_analysis.py](ems_whisper_eval_bundle/run_gap_analysis.py)（產出 §5 的 DSP + encoder 距離）
- B_1ep 訓練輸出：[speaker_aug_sweep_clean/B_clean_20spk_orig_1ep/](speaker_aug_sweep_clean/B_clean_20spk_orig_1ep/)、CSV [B_clean_20spk_orig_1ep_on_aligned.csv](speaker_aug_sweep_clean/B_clean_20spk_orig_1ep_on_aligned.csv)
- VB gap 分析結果：[ems_whisper_eval_bundle/vb_gap_results/](ems_whisper_eval_bundle/vb_gap_results/)（gap_stats.csv / gap_distances.csv / gap_tsne.png）
- Sweep runner：[run_speaker_aug_sweep_clean.sh](run_speaker_aug_sweep_clean.sh)
- Guarded inference：[run_whisper_inference_guarded.py](run_whisper_inference_guarded.py)
- Broadcastify seed builder：[build_broadcastify_seeds.py](build_broadcastify_seeds.py)
