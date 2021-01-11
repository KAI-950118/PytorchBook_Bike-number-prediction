import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
# %matplotlib inline # cannot use in Pycharm

data_path = 'hour.csv'
rides = pd.read_csv(data_path)
rides.head()
counts = rides['cnt'][:50]
# x = np.arange(len(counts), dtype=float)
# y = np.array(counts, dtype=float)
# # row data print
# # plt.figure(figsize=(10, 7))
# # plt.plot(x, y, 'o-')
# # plt.xlabel("X")
# # plt.ylabel("Y")
# # plt.show()
#
# # fail method
# X = Variable(torch.FloatTensor(x)/len(counts))
# Y = Variable(torch.FloatTensor(y))
#
# sz = 10
# weights = Variable(torch.randn(1, sz), requires_grad=True)
# biases = Variable(torch.randn(sz), requires_grad=True)
# weights_2 = Variable(torch.randn(sz, 1), requires_grad=True)
# learning_rate = 0.01
# losses = []
#
# for i in range(400000):
#     hidden = X.expand(sz, len(x)).t() * weights.expand(len(x), sz) + biases.expand(len(x), sz)
#     hidden = torch.sigmoid(hidden)
#     predictions = hidden.mm(weights_2)
#     loss = torch.mean((predictions - Y) ** 2)
#     losses.append(loss.data.numpy())
#
#     if i % 10000 == 0:
#         print("loss:", loss)
#
#     loss.backward()
#     weights.data.add_(-learning_rate * weights.grad.data)
#     biases.data.add_(-learning_rate * biases.grad.data)
#     weights_2.data.add_(-learning_rate * weights_2.grad.data)
#
#     weights.grad.data.zero_()
#     biases.grad.data.zero_()
#     weights_2.grad.data.zero_()
#
# # print("finish")
# # plt.plot(losses)
# # plt.xlabel("Epoch")
# # plt.ylabel("Loss")
# # plt.show()
#
# plt.figure(figsize=(10, 7))
# xplot,= plt.plot(X.data.numpy(), Y.data.numpy(), 'o')
# yplot,= plt.plot(X.data.numpy(), predictions.data.numpy())
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend([xplot, yplot], ['Data', "predictions"])
# plt.show()

## Neural Network
dummy_feilds = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
# print(rides)
for each in dummy_feilds:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)
# print(rides)
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
# print(data)
quant_feature = ['cnt', 'temp', 'hum', 'windspeed']
scaled_feature = {}
for each in quant_feature:
    mean, std = data[each].mean(), data[each].std()
    scaled_feature[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
# print(data)
test_data = data[-21*24:]
train_data = data[:-21*24]

target_fields = ['cnt', 'casual', 'registered']

feature, targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
test_feature, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

X = feature.values
Y = targets['cnt'].values
# print(Y)
Y = Y.astype(float)
# print(Y)
Y = np.reshape(Y, [len(Y), 1]) #Trans to Y*1 matrix
# print(Y)
losses = []

input_size = feature.shape[1]
hidden_size = 10
output_size = 1
batch_size = 128

neu = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size),
)
cost = torch.nn.MSELoss()

optimizer = torch.optim.SGD(neu.parameters(), lr=0.01)
for i in range(1000):
    batch_loss = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size if start + batch_size < len(X) else len(X)
        xx = Variable(torch.FloatTensor(X[start:end]))
        yy = Variable(torch.FloatTensor(Y[start:end]))
        predict = neu(xx)
        loss = cost(predict, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.data.numpy())
    if i % 100==0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

plt.plot(np.arange(len(losses))*100, losses)
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.show()

targets = test_targets['cnt']
targets = targets.values.reshape([len(targets), 1])
targets = targets.astype(float)

xxx = Variable(torch.FloatTensor(test_feature.values))
yyy = Variable(torch.FloatTensor(targets))

predict = neu(xxx)
predict = predict.data.numpy()

fig, ax = plt.subplots(figsize=(10, 7))
mean, std = scaled_feature['cnt']
ax.plot(predict * std + mean, label='Prediction')
ax.plot(targets * std + mean, label='Data')
ax.legend()
ax.set_xlabel('Date-time')
ax.set_ylabel('Counts')
dates = pd.to_datetime(rides.loc[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
# print(dates)
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
plt.show()

def feature(X, net):
    X = Variable(torch.from_numpy(X).type(torch.FloatTensor), requires_grad=False)
    dic = dict(net.named_parameters())
    weights = dic["0.weight"]
    biases = dic['0.bias']
    h = torch.sigmoid(X.mm(weights.t())+biases.expand(len(X), len(biases)))
    return h

bool1 = rides['dteday'] == '2012-12-22'
bool2 = rides['dteday'] == '2012-12-23'
bool3 = rides['dteday'] == '2012-12-24'
bools = [any(tup) for tup in zip(bool1, bool2, bool3)]


subset = test_feature.loc[rides[bools].index]
subtargets = test_targets.loc[rides[bools].index]
subtargets = subtargets['cnt']
subtargets = subtargets.values.reshape([len(subtargets), 1])

results = feature(subset.values, neu).data.numpy()
predict = neu(Variable(torch.FloatTensor(subset.values))).data.numpy()
mean, std = scaled_feature['cnt']
predict = predict * std + mean
subtargets = subtargets * std + mean

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(results[:, :], '.:', alpha=0.1)
ax.plot((predict - min(predict)) / (max(predict) - min(predict)), 'bo-', label='Prediction')
ax.plot((subtargets - min(predict)) / (max(predict) - min(predict)), 'ro-', label='Real')
ax.plot(results[:, 6], '.:', alpha=1, label='Neuro 7')
ax.set_xlim(right=len(predict))
ax.legend()
plt.ylabel('Normalize Values')
dates = pd.to_datetime(rides.loc[subset.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
plt.show()

dic = dict(neu.named_parameters())
weights = dic['0.weight']
plt.plot(weights.data.numpy()[4, :], 'o-')
plt.xlabel('input Neurons')
plt.ylabel('Weight')
plt.show()