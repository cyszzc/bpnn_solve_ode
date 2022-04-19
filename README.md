# 深度神经网络求解常微分方程
### 说明
<br>``BP_ODE.cpp``为单隐含层bp神经网络模型
<br>``ODE_BPx3.cpp``为三层隐含层bp神经网络模型
<br>``testdata.txt``为算例一的训练样本，``traindata.txt``为算例一的测试用样本
<br>``testdata_t2.txt``为算例二的训练样本，``traindata_t2.txt``为算例二的测试用样本
### 算例介绍
算例一<br>
![image](https://user-images.githubusercontent.com/61587007/163974992-800ed7fc-56c3-47d3-9b69-00eebea9374c.png)
<br>算例二<br>
![image](https://user-images.githubusercontent.com/61587007/163975058-67b48965-ed1a-4a17-9b17-dfb003eb3daf.png)
### 神经网络模型
**单隐含层bp神经网络模型**<br>
.<div align=center></div>![image](https://user-images.githubusercontent.com/61587007/163971587-487b116c-fdb8-41f9-bc2d-c912f5ab4cf2.png)
<br>**三层隐含层bp神经网络模型**<br>
.<div align=center></div>![image](https://user-images.githubusercontent.com/61587007/163971745-3dd499a7-ec38-4ec7-8794-c49e8e24e855.png)
### 其他
可以通过修改代码中的``mosttimes``变量的值来更改最大训练次数
<br>修改``INNODE``、``HIDENODE``、``OUTNODE``的值来更改神经元数量
<br>修改``rate``的值来更改学习率的值
<br>项目中的``.txt``文件可能在Windows环境下会出现读入数据多一个空行，导致程序报错，需要检查文件最后是否多出一个空行
