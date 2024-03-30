clear;close all;clc
%% Load Data
data = readmatrix('total.csv');
data = data(2:end,:);
%% Define regression method and decomposition method
regression_method = 'SVM';% 'SVM', 'LSBoost', 'LightGBM', 'XGBoost','BiLSTM'
decomposition_method = 'VMD'; % 'EEMD','VMD,'EFD','none'
spilt_ri = [6 2 2]; % The ratio of training, validation and testing set
%% Decomposition method
tic
t = 1:length(data(:,end));
deo_num = 11;
switch decomposition_method
    case 'none'
        data_cell{1,1} = data;
    case 'VMD'
        data_cell=[];
        [imf,res] = vmd(data(:,end),'NumIMF',deo_num);
        for i=1:deo_num
            data_cell{1,i} = [data(:,1:end-1),imf(:,i)];
        end
    case 'EFD'
        imf = EFD(data(:,end),deo_num);
        for i=1:deo_num
            data_cell{1,i} = [data(:,1:end-1),cell2mat(imf(i))'];
        end
    case 'EEMD'
        Nstd=0.25;    NE=50;%  Nstd: ratio of the standard deviation of the added noise and that of Y;NE: Ensemble number for the EEMD
        IMF = eemd(data(:,end),Nstd,NE,deo_num);
        imf=IMF(:,2:end);
        for i=1:deo_num+1
            data_cell{1,i} = [data(:,1:end-1),imf(:,i)];
        end
end
%% Training models
x_mu_all=[]; x_sig_all=[]; y_mu_all=[]; y_sig_all=[];
for NUM_all=1:length(data_cell)
    data_process = data_cell{1,NUM_all};
    [x_feature_label,y_feature_label] = timeseries_process(data_process,1,2,10);
    [~,y_feature_label1] = timeseries_process(data,1,2,10);    %未分解之前

    index_label1=1:(size(x_feature_label,1)); index_label=index_label1;
    train_num = round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));                    %训练集个数
    vaild_num = round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); %验证集个数
    % 训练集，验证集，测试集
    train_x_feature_label = x_feature_label(index_label(1:train_num),:);
    train_y_feature_label = y_feature_label(index_label(1:train_num),:);
    vaild_x_feature_label = x_feature_label(index_label(train_num+1:vaild_num),:);
    vaild_y_feature_label = y_feature_label(index_label(train_num+1:vaild_num),:);
    test_x_feature_label  = x_feature_label(index_label(vaild_num+1:end),:);
    test_y_feature_label  = y_feature_label(index_label(vaild_num+1:end),:);
    % Zscore 标准化

    x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label);
    train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;  % 训练数据标准化
    y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label);
    train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;  % 训练数据标准化
    x_mu_all(NUM_all,:)=x_mu;x_sig_all(NUM_all,:)=x_sig;y_mu_all(NUM_all,:)=y_mu;y_sig_all(NUM_all,:)=y_sig;

    vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;  % 训练数据标准化
    vaild_y_feature_label_norm = (vaild_y_feature_label - y_mu) ./ y_sig;  % 训练数据标准化

    test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化
    test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化

    % 创建空的数组存储预测数据
    y_train_predict_norm = zeros(size(train_y_feature_label,1),size(train_y_feature_label,2));y_vaild_predict_norm=zeros(size(vaild_y_feature_label,1),size(vaild_y_feature_label,2));
    y_test_predict_norm = zeros(size(test_y_feature_label,1),size(test_y_feature_label,2));

    switch regression_method
        case 'RF'
            Mdl= TreeBagger(100,train_x_feature_label_norm,train_y_feature_label_norm(:,1),'Method','regression');
            y_train_predict_norm(:,1) = predict(Mdl,train_x_feature_label_norm);
            y_vaild_predict_norm(:,1) = predict(Mdl,vaild_x_feature_label_norm);
            y_test_predict_norm(:,1) = predict(Mdl,test_x_feature_label_norm);
        case 'SVM'
            Mdl = fitrsvm(train_x_feature_label_norm,train_y_feature_label_norm(:,1));
            y_train_predict_norm(:,1) = predict(Mdl,train_x_feature_label_norm);
            y_vaild_predict_norm(:,1) = predict(Mdl,vaild_x_feature_label_norm);
            y_test_predict_norm(:,1) = predict(Mdl,test_x_feature_label_norm);
        case 'LSBoost'
            Mdl= fitrensemble(train_x_feature_label_norm,train_y_feature_label_norm(:,1));
            y_train_predict_norm(:,1) = predict(Mdl,train_x_feature_label_norm);
            y_vaild_predict_norm(:,1) = predict(Mdl,vaild_x_feature_label_norm);
            y_test_predict_norm(:,1) = predict(Mdl,test_x_feature_label_norm);
        case 'XGBoost'
            paramters.maxiter = 50;        %最大迭代次数
            paramters.train_booster = 'gbtree';
            paramters.objective = 'reg:linear';
            paramters.depth_max = 5;    %最大深度
            paramters.learn_rate = 0.1;   %学习率
            paramters.min_child = 1;      %最小叶子
            paramters.subsample = 0.95;  %采样
            paramters.colsample_bytree = 1;
            paramters.num_parallel_tree = 1;
            Mdl = train_xgb(train_x_feature_label_norm, train_y_feature_label_norm(:,1), paramters);

            y_train_predict_norm(:,1) = predict_xgb(Mdl,train_x_feature_label_norm);
            y_vaild_predict_norm(:,1) = predict_xgb(Mdl,vaild_x_feature_label_norm);
            y_test_predict_norm(:,1) = predict_xgb(Mdl,test_x_feature_label_norm);
        case 'LightGBM'
            if not(libisloaded('lib_liblightgbm'))
                loadlibrary('lib_lightgbm.dll','c_api.h')
            end
            parameters = containers.Map;
            parameters('verbose') = 1;              % 可视化训练信息
            parameters('task') = 'train';           % 训练模型
            parameters('num_threads') = 1;          % 阈值?
            parameters('num_leaves') = 6;           % 叶子节点数
            parameters('bagging_freq') = 8;         % 建树的样本采样比例
            parameters('metric') = 'rmse';          % 评价指标
            parameters('learning_rate') = 0.06;     % 学习率
            parameters('boosting_type') = 'gbdt';   % 设置提升类型
            parameters('feature_fraction') = 0.9;   % 建树的特征选择比例
            parameters('bagging_fraction') = 0.7;   % 建树的样本采样比例
            parameters('objective') = 'regression'; % 回归任务

            pv_train = lgbmDataset(train_x_feature_label_norm);
            setField(pv_train, 'label', train_y_feature_label_norm);

            pv_valid = lgbmDataset(vaild_x_feature_label_norm,pv_train);
            setField(pv_valid, 'label', vaild_y_feature_label_norm);

            pv_test = lgbmDataset(test_x_feature_label_norm);
            setField(pv_test, 'label', test_y_feature_label_norm);

            num_boost_round = 300;                   % 最大迭代次数
            early_stopping_rounds = 50;              % 早停参数
            [booster, bestIteration, metrics, metricNames] = train(pv_train, parameters, num_boost_round, pv_valid, early_stopping_rounds);

            y_train_predict_norm(:,1) = booster.predictMatrix(train_x_feature_label_norm , bestIteration);
            y_vaild_predict_norm(:,1) = booster.predictMatrix(vaild_x_feature_label_norm , bestIteration);
            y_test_predict_norm(:,1) = booster.predictMatrix(test_x_feature_label_norm , bestIteration);
        case 'BiLSTM'            
            p_train1=cell(size(train_x_feature_label,1),1);p_test1=cell(size(test_x_feature_label,1),1);p_vaild1=cell(size(vaild_x_feature_label,1),1);
            O_train1=cell(size(train_x_feature_label,1),1);O_test1=cell(size(test_x_feature_label,1),1);O_vaild1=cell(size(vaild_x_feature_label,1),1);
            for i = 1: size(train_x_feature_label,1)      %修改输入变成元胞形式
                p_train1{i, 1} = (train_x_feature_label_norm(i,:))';
            end
            for i = 1 : size(test_x_feature_label,1)
                p_test1{i, 1}  = (test_x_feature_label_norm(i,:))';
            end
            for i = 1 : size(vaild_x_feature_label,1)
                p_vaild1{i, 1}  = (vaild_x_feature_label_norm(i,:))';
            end

            for i = 1: size(train_x_feature_label,1)      %修改输入变成元胞形式
                O_train1{i, 1} = (train_y_feature_label_norm(i,1))';
            end
            for i = 1 : size(test_x_feature_label,1)
                O_test1{i, 1}  = (test_y_feature_label_norm(i,1))';
            end
            for i = 1 : size(vaild_x_feature_label,1)
                O_vaild1{i, 1}  = (vaild_y_feature_label_norm(i,1))';
            end

            hidden_size=64;
            if(length(hidden_size)<2)
                layers = [sequenceInputLayer(size(train_x_feature_label,2))
                    bilstmLayer(hidden_size(1), 'OutputMode', 'sequence')      % LSTM层
                    reluLayer                                               % Relu激活层
                    dropoutLayer(0.2)                                 % 防止过拟合
                    fullyConnectedLayer(size(train_y_feature_label(:,1),2))          % 全连接层
                    regressionLayer];

            elseif (length(hidden_size)>=2)
                layers = [sequenceInputLayer(size(train_x_feature_label,2))
                    bilstmLayer(hidden_size(1),'OutputMode','sequence')
                    dropoutLayer(0.2)
                    bilstmLayer(hidden_size(2),'OutputMode','sequence')
                    dropoutLayer(0.2)
                    fullyConnectedLayer(size(train_y_feature_label(:,1),2))
                    regressionLayer];
            end
            batchsize = 256;
            options = trainingOptions('adam', ...
                'MaxEpochs',100, ...
                'MiniBatchSize',batchsize,...
                'InitialLearnRate',0.001,...
                'ValidationFrequency',20, ...
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropPeriod',125, ...
                'LearnRateDropFactor',0.2, ...
                'Plots','training-progress');
            Mdl = trainNetwork(p_train1, O_train1, layers, options);
            y_train_predict_norm1 = predict(Mdl, p_train1,'MiniBatchSize',batchsize);
            y_vaild_predict_norm1 = predict(Mdl, p_vaild1,'MiniBatchSize',batchsize);
            y_test_predict_norm1 =  predict(Mdl, p_test1,'MiniBatchSize',batchsize);
            y_train_predict_norm_roll=[];
            y_vaild_predict_norm_roll=[];
            y_test_predict_norm_roll=[];

            for i=1:length(y_train_predict_norm1)
                y_train_predict_norm_roll(i,:) = (y_train_predict_norm1{i,1});
            end
            for i=1:length(y_vaild_predict_norm1)
                y_vaild_predict_norm_roll(i,:) = (y_vaild_predict_norm1{i,1});
            end
            for i=1:length(y_test_predict_norm1)
                y_test_predict_norm_roll(i,:) = (y_test_predict_norm1{i,1});
            end
            y_train_predict_norm(:,1) = y_train_predict_norm_roll;
            y_vaild_predict_norm(:,1) = y_vaild_predict_norm_roll;
            y_test_predict_norm(:,1) = y_test_predict_norm_roll;
    end

    y_train_predict_cell{1,NUM_all} = y_train_predict_norm.*y_sig + y_mu;  % 反标准化操作
    y_vaild_predict_cell{1,NUM_all} = y_vaild_predict_norm.*y_sig + y_mu;
    y_test_predict_cell{1,NUM_all} = y_test_predict_norm.*y_sig + y_mu;
end

y_train_predict = 0; y_vaild_predict = 0; y_test_predict = 0;
for i = 1:length(data_cell)
    y_train_predict = y_train_predict + y_train_predict_cell{1,i};
    y_vaild_predict = y_vaild_predict + y_vaild_predict_cell{1,i};
    y_test_predict = y_test_predict + y_test_predict_cell{1,i};
end

train_y_feature_label = y_feature_label1(index_label(1:train_num),:);
vaild_y_feature_label = y_feature_label1(index_label(train_num+1:vaild_num),:);
test_y_feature_label = y_feature_label1(index_label(vaild_num+1:end),:);
%% Calculating metrics
Tvalue = [regression_method,'_',decomposition_method];
train_y = train_y_feature_label;
train_MAE = sum(sum(abs(y_train_predict-train_y)))/size(train_y,1)/size(train_y,2) ; disp([Tvalue,'训练集平均绝对误差MAE：',num2str(train_MAE)])
train_MAPE = sum(sum(abs((y_train_predict-train_y)./train_y)))/size(train_y,1)/size(train_y,2); disp([Tvalue,'训练集平均相对误差MAPE：',num2str(train_MAPE)])
train_MSE = (sum(sum(((y_train_predict-train_y)).^2))/size(train_y,1)/size(train_y,2)); disp([Tvalue,'训练集均方根误差MSE：',num2str(train_MSE)])
train_RMSE = sqrt(sum(sum(((y_train_predict-train_y)).^2))/size(train_y,1)/size(train_y,2)); disp([Tvalue,'训练集均方根误差RMSE：',num2str(train_RMSE)])
train_R2 = 1 - mean(norm(train_y - y_train_predict)^2 / norm(train_y - mean(train_y))^2);   disp([Tvalue,'训练集均方根误差R2：',num2str(train_R2)])
disp('************************************************************************************')
vaild_y = vaild_y_feature_label;
vaild_MAE = sum(sum(abs(y_vaild_predict-vaild_y)))/size(vaild_y,1)/size(vaild_y,2) ; disp([Tvalue,'验证集平均绝对误差MAE：',num2str(vaild_MAE)])
vaild_MAPE = sum(sum(abs((y_vaild_predict-vaild_y)./vaild_y)))/size(vaild_y,1)/size(vaild_y,2); disp([Tvalue,'验证集平均相对误差MAPE：',num2str(vaild_MAPE)])
vaild_MSE = (sum(sum(((y_vaild_predict-vaild_y)).^2))/size(vaild_y,1)/size(vaild_y,2)); disp([Tvalue,'验证集均方根误差MSE：',num2str(vaild_MSE)])
vaild_RMSE = sqrt(sum(sum(((y_vaild_predict-vaild_y)).^2))/size(vaild_y,1)/size(vaild_y,2)); disp([Tvalue,'验证集均方根误差RMSE：',num2str(vaild_RMSE)])
vaild_R2 = 1 - mean(norm(vaild_y - y_vaild_predict)^2 / norm(vaild_y - mean(vaild_y))^2);   disp([Tvalue,'验证集均方根误差R2：',num2str(vaild_R2)])
disp('************************************************************************************')
test_y = test_y_feature_label;
test_MAE = sum(sum(abs(y_test_predict-test_y)))/size(test_y,1)/size(test_y,2) ; disp([Tvalue,'测试集平均绝对误差MAE：',num2str(test_MAE)])
test_MAPE = sum(sum(abs((y_test_predict-test_y)./test_y)))/size(test_y,1)/size(test_y,2); disp([Tvalue,'测试集平均相对误差MAPE：',num2str(test_MAPE)])
test_MSE = (sum(sum(((y_test_predict-test_y)).^2))/size(test_y,1)/size(test_y,2)); disp([Tvalue,'测试集均方根误差MSE：',num2str(test_MSE)])
test_RMSE = sqrt(sum(sum(((y_test_predict-test_y)).^2))/size(test_y,1)/size(test_y,2)); disp([Tvalue,'测试集均方根误差RMSE：',num2str(test_RMSE)])
test_R2 = 1 - mean(norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);   disp([Tvalue,'测试集均方根误差R2：',num2str(test_R2)])
toc
%% Plotting results
num_plot = round(0.2*size(data,1))-10;
figure(1)
tiledlayout(1,3,"TileSpacing","tight","Padding","tight")
set(gcf,'Position',[392,479,1186,340],'Units','pixels')
nexttile
scatter(train_y,y_train_predict,'filled')
xlabel('True')
ylabel('Pred')
set(gca,'FontSize',16)

nexttile
scatter(vaild_y,y_vaild_predict,'filled')
xlabel('True')
ylabel('Pred')
set(gca,'FontSize',16)

nexttile
scatter(test_y,y_test_predict,'filled')
xlabel('True')
ylabel('Pred')
set(gca,'FontSize',16)

figure(2)
tiledlayout(1,3,"TileSpacing","tight","Padding","tight")
set(gcf,'Position',[392,479,1186,340],'Units','pixels')
nexttile
plot(train_y(1:num_plot),'LineWidth',2)
hold on 
plot(y_train_predict(1:num_plot),'-*')

nexttile
plot(vaild_y(1:num_plot),'LineWidth',2)
hold on 
plot(y_vaild_predict(1:num_plot),'-*')

nexttile
plot(test_y(1:num_plot),'LineWidth',2)
hold on 
plot(y_test_predict(1:num_plot),'-*')
%%
% train_table = table(train_y, y_train_predict, 'VariableNames',{'train_y','predict'});
% valid_table = table(vaild_y, y_vaild_predict, 'VariableNames',{'valid_y','predict'});
% test_table = table(test_y,y_test_predict, 'VariableNames',{'test_y','predict'});
% metrics = table(train_RMSE,train_R2,vaild_RMSE,vaild_R2,test_RMSE,test_R2,'VariableNames',{'train_RMSE','train_R2','vaild_RMSE','vaild_R2','test_RMSE','test_R2'});
% filename = [Tvalue,num2str(deo_num),'.xlsx'];
% writetable(train_table, filename, 'Sheet', 'Sheet1');
% writetable(valid_table, filename, 'Sheet', 'Sheet2');
% writetable(test_table, filename, 'Sheet', 'Sheet3');
% writetable(metrics,filename,'Sheet', 'Sheet4');