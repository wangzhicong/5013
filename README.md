# log 
## 1005
* 修改了预测方式，从预测下一个时间点改为预测之后第60*3个时间点（或更多）
* 修改了购买方式，每次只够买固定的量。增加了追买追卖功能，如果预测时间点涨幅超过阈值，那么直到预测时间点之前所有预测价格比现在时间点低的时候都买入

### 收益
* 1 - 9：0.05965598199662381
* 9 - 16：-0.0119
* 16 - 23：-0.050679914507123724
* 23- 30：

## 1006
重构代码，感谢kuojian