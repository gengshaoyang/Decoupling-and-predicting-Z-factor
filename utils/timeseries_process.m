function [x_feature_label,y_feature_label]=timeseries_process(data_select,select_predict_num,num_feature,num_series)
% 可知未来特征的多元时间序列预测
% select_predict_num=str2double(app.EditField_31.Value);    %特征选择个数
% num_features=2*select_predict_num;
num_train=length(data_select)-num_series;
for i=1:num_train-select_predict_num+1
  timefeaturedata= data_select(i:i+num_series-1,end);
  feature_select=data_select(i+num_series-num_feature:i+num_series-1,1:end-1);
  net_input(i,:)=[feature_select(:)',timefeaturedata(:)'];
end


for i=1:num_train-select_predict_num+1
  timelabel= data_select(i+num_series:i+num_series+select_predict_num-1,end);
  net_output(i,:)=timelabel(:)';    
  feature_select2= data_select(i+num_series:i+num_series+select_predict_num-1,1:end-1);
  net_input1(i,:)=feature_select2(:)';
end
  net_input2=[net_input1,net_input];
  x_feature_label=net_input2;
  y_feature_label=net_output;
end