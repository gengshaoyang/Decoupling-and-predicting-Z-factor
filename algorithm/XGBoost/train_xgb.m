function model = train_xgb(X_train, y_train, paramters)
%%  Function inputs:
%  x_train:        matrix of inputs for the training set
%  y_train:        vetor of labels/values for the test set
%  params :        structure of learning parameters

%%  Function output:
%  model: a structure containing:
%  iters_optimal; % number of iterations performs by xgboost (final model)
%  h_booster_ptr; % pointer to the final model
%  params;        % model parameters (just for info)
%  missing;       % value considered "missing"

%%  ���� xgboost ��
if not(libisloaded('xgboost'))
    loadlibrary('xgboost.dll','xgboost.h')
end
max_num_iters=paramters.maxiter;

%%  ���ò���
missing = single(NaN);          % ���ø�ֵ����Ϊ"ȱʧ"
iters_optimal = max_num_iters;  % ����������

%%  ����xgboost����ز���
if isempty(paramters)
    params.booster           = 'gbtree';
    % params.objective         = 'binary:logistic';
    params.objective         = 'reg:linear';
    params.max_depth         = 5;
    params.eta               = 0.1;
    params.min_child_weight  = 1;
    params.subsample         = 0.9;
    params.colsample_bytree  = 1;
    params.num_parallel_tree = 1;
else
    params.booster   =paramters.train_booster;
    % params.objective         = 'binary:logistic';
    params.objective    = paramters.objective;
    params.max_depth         = paramters.depth_max;
    params.eta               = paramters.learn_rate;
    params.min_child_weight  = paramters.min_child;
    params.subsample         = paramters.subsample;
    params.colsample_bytree  = paramters.colsample_bytree;
    params.num_parallel_tree = paramters.num_parallel_tree;   
end

%%  ������ת��Ϊȫ������
param_fields = fields(params);
for i = 1 : length(param_fields)
    eval(['params.' param_fields{i} ' = num2str(params.' param_fields{i} ');'])
end

%%  �õ����������������
rows = uint64(size(X_train, 1));  % �������ݵ��� ������
cols = uint64(size(X_train, 2));  % �������ݵ��� ������
X_train = X_train';

%%  �������ָ��
p_train_ptr = libpointer('singlePtr', single(X_train));
t_train_ptr = libpointer('singlePtr', single(y_train));

h_train_ptr = libpointer;
h_train_ptr_ptr = libpointer('voidPtrPtr', h_train_ptr);

%%  ������������
calllib('xgboost', 'XGDMatrixCreateFromMat', p_train_ptr, rows, cols, missing, h_train_ptr_ptr);

%%  �����ǩ
labelStr = 'label';
calllib('xgboost', 'XGDMatrixSetFloatInfo', h_train_ptr, labelStr, t_train_ptr, rows);

%%  ���������������ò���
h_booster_ptr = libpointer;
h_booster_ptr_ptr = libpointer('voidPtrPtr', h_booster_ptr);
calllib('xgboost', 'XGBoosterCreate', h_train_ptr_ptr, uint64(1), h_booster_ptr_ptr);

for i = 1 : length(param_fields)
    eval(['calllib(''xgboost'', ''XGBoosterSetParam'', h_booster_ptr, ''' param_fields{i} ''', ''' eval(['params.' param_fields{i}]) ''');'])
end

%%  ����ģ��
for iter = 0 : iters_optimal
    calllib('xgboost', 'XGBoosterUpdateOneIter', h_booster_ptr, int32(iter), h_train_ptr);
end

%%  ��ģ�Ͳ������浽 model
model                = struct;
model.iters_optimal  = iters_optimal;  % ����������
model.h_booster_ptr  = h_booster_ptr;  % ָ������ģ�͵�ָ��
model.params         = params;         % ��ز���
model.missing        = missing;        % ȱʧֵ
