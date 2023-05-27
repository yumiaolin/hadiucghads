#导入所需的包
import numpy as np
from metrics.ETTh_metrics import metric
import torch
from sklearn.preprocessing import StandardScaler

# #导入npy文件路径位置
# test = np.load('./exp/ett_results/SCINet_FJT_S_ftS_sl96_ll48_pl48_lr0.003_bs8_hid4.0_s1_l3_dp0.5_invFalse_itr0/pred.npy')
#
# print(test)

# def predict_from_IMFs(imfs):
#     n = len(imfs)
#     assert n > 0, "Invalid input: empty list!"
#     assert all([imfs[i].shape == imfs[0].shape for i in range(n)]), "Invalid input: IMF shapes do not match!"
#
#     n_samples = imfs[0].shape[0]
#     final_pred = torch.zeros(n_samples)
#
#     for i in range(n):
#         cur_pred = torch.zeros(n_samples)
#         for j in range(i + 1):
#             cur_pred += imfs[j]
#         final_pred = final_pred.reshape(1, n_samples)
#         final_pred += cur_pred
#
#     return final_pred
#VMD分解后归一化预测值和真实值读取
pred1= np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF1/pred.npy')
pred2= np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF2/pred.npy')
pred3= np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF3/pred.npy')
pred4= np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF4/pred.npy')
true1=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF1/true.npy')
true2=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF2/true.npy')
true3=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF3/true.npy')
true4=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF4/true.npy')

#VMD分解后反归一化预测值和真实值读取
true_scales1=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF1/true_scales.npy')
true_scales2=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF2/true_scales.npy')
true_scales3=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF3/true_scales.npy')
true_scales4=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF4/true_scales.npy')
pred_scales1=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF1/pred_scales.npy')
pred_scales2=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF2/pred_scales.npy')
pred_scales3=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF3/pred_scales.npy')
pred_scales4=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/IMF4/pred_scales.npy')

#未作VMD反归一化预测值和真实值读取
pred_scales5=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/跑FJT_S/pred_scales.npy')
true_scales5=np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/跑FJT_S/true_scales.npy')
#未作VMD归一化预测值和真实值读取
pred5= np.load('D:/桌面/研究生阶段事宜/研一/风功率预测/SCINet跑FJT_S分解VMD/跑FJT_S/pred.npy')
#metric=np.load('./exp/ett_results/SCINet_FJT_S_ftS_sl96_ll48_pl48_lr0.003_bs8_hid4.0_s1_l3_dp0.5_invTrue_itr0/metrics.npy')
#FJT_S不做VMD分解
# pred=np.load('./exp/ett_results/SCINet_FJT_S_ftS_sl96_ll48_pl48_lr0.003_bs8_hid4.0_s1_l3_dp0.5_invFalse_itr0/pred.csv.npy')
# true=np.load('./exp/ett_results/SCINet_FJT_S_ftS_sl96_ll48_pl48_lr0.003_bs8_hid4.0_s1_l3_dp0.5_invFalse_itr0/true.csv.npy')
# imfs=[]
# imfs.append(pred1)
# imfs.append(pred2)
# imfs.append(pred3)
# imfs.append(pred4)
# final_pred = predict_from_IMFs(imfs)
# n = len(pred1)  # 时间序列的长度（假设所有 IMF 预测结果的长度相同
# final_pred = torch.zeros(n,48,n)  # 初始化最终预测结果为全0数组

# for i in range(4):
#     cur_pred = torch.zeros(n)  # 初始化当前 IMF 的预测结果为全0数组
#     for j in range(i+1):  # 将所有高频 IMF 的预测结果累加到当前 IMF 的预测结果上
#         cur_pred += locals()["pred"+str(j+1)]  # 使用locals()函数获取对应 IMF 的变量
#     final_pred += cur_pred  # 将当前 IMF 的预测结果加到最终预测结果上

    # scaler =StandardScaler()
#     scaler.fit(final_pred)
#     data = scaler.transform(final_pred)
# predict_value = mm.inverse_transform(final_pred)
#y_pred = scaler.inverse_transform(final_pred)
final_true=true1+true2+true3+true4
final_pred=pred1+pred2+pred3+pred4

pred_scales=pred_scales2+pred_scales4+pred_scales3+pred_scales1
true_scales=true_scales1+true_scales3+true_scales4+true_scales2
mae, mse, rmse, mape, mspe, corr = metric(pred_scales5, true_scales5)
print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape,mspe, corr))
# mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
