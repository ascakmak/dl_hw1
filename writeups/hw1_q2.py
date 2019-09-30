import numpy as np

def forward(w):
    w1 = w[0]
    w2 = w[1]
    temp = np.exp(w1)+np.exp(2*w2)
    f1 = np.exp(temp) + np.sin(temp)
    f2 = w1*w2 + sigmoid(w1)
    return np.array([f1,f2])

def forward_auto(w):
    w1 = w[0]
    w2 = w[1]
    dw1 = 1
    dw2 = 1
    temp = np.exp(w1)+np.exp(2*w2)
    dtemp_dw1 = np.exp(w1)*dw1
    dtemp_dw2 =  2*np.exp(2*w2)*dw2
    f = np.zeros((2,1))
    f[0] = np.exp(temp) + np.sin(temp)
    f[1] = w1*w2 + sigmoid(w1)
    df_dw = np.zeros((2,2))
    df_dw[0,0] = np.exp(temp)*dtemp_dw1 + np.cos(temp)*dtemp_dw1
    df_dw[1,0] = np.exp(temp)*dtemp_dw2 + np.cos(temp)*dtemp_dw2
    df_dw[0,1] = dw1*w2 + sigmoid(w1)*(1-sigmoid(w1))*dw1
    df_dw[1,1] = w1*dw2
    return (f, df_dw)


def backward_auto(w):
    w1 = w[0]
    w2 = w[1]
    dw1 = 1
    dw2 = 1
    temp = np.exp(w1)+np.exp(2*w2)
    f = np.zeros((2,1))
    f[0] = np.exp(temp) + np.sin(temp)
    f[1] = w1*w2 + sigmoid(w1)

    df1_dtemp = np.exp(temp) + np.cos(temp)
    df1_w1 = df1_dtemp * np.exp(w1)
    df1_w2 = df1_dtemp * 2 * np.exp(2*w2)
    df_dw[0,0] = df1_w1
    df_dw[1,0] = df1_w2
    df_dw[0,1] = w2 + sigmoid(w1)*(1-sigmoid(w1))
    df_dw[1,1] = w1
    return (f, df_dw)

def sigmoid(x):
    y = 1/(1+np.exp(-x));
    return y

# Q2.a
w = np.array([1,2])
print("Forward Pass at w=[1,2]: ")
print(forward(w))
# Q2.b
print("Numerical Differentation: ")
delta_w1 = np.array([0.1, 0])
delta_w2 = np.array([0, 0.1])
print((forward(w+delta_w1)-forward(w-delta_w1))/(2*0.1))
print((forward(w+delta_w2)-forward(w-delta_w2))/(2*0.1))
# Q2.c
print("Forward Auto Differentation: ")
f, df_dw = forward_auto(w)
print(df_dw)
# Q2.d
print("Backward Auto Differentation: ")
f, df_dw = backward_auto(w)
print(df_dw)
