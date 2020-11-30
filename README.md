# TOSEM Revision

## Notes

* Three datasets: training dataset, validation dataset, and test dataset.
* split 20% samples from the training dataset.

### 命名规则
* `weights dir`存放路径：`./weights/<model name>/<dataset>/<version>/`
* `.h5`文件命名：
    * `trained.h5`直接训练完后的模型
    * `apricot_fixed_<strategy>.h5`使用`Apricot`之后的模型
    * `plus_fixed_<strategy>.h5`使用`Apricot Plus`之后的模型
    * `lite_fixed.h5`使用`Apricot Lite`之后的模型
