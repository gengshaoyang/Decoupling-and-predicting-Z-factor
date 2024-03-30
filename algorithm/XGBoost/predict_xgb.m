function Ypre =predict_xgb(model,p_test)

%%  ��ȡģ��
h_booster_ptr = model.h_booster_ptr;

%%  �õ����������������
rows = uint64(size(p_test, 1));
cols = uint64(size(p_test, 2));
p_test = p_test'; 

%%  ���ñ�Ҫ��ָ��
h_test_ptr = libpointer;
h_test_ptr_ptr = libpointer('voidPtrPtr', h_test_ptr);
test_ptr = libpointer('singlePtr', single(p_test));
calllib('xgboost', 'XGDMatrixCreateFromMat', test_ptr, rows, cols, model.missing, h_test_ptr_ptr);

%%  ����Ԥ��
out_len_ptr = libpointer('uint64Ptr', uint64(0));
f = libpointer('singlePtr');
f_ptr = libpointer('singlePtrPtr', f);
calllib('xgboost', 'XGBoosterPredict', h_booster_ptr, h_test_ptr, int32(0), uint32(0), int32(0), out_len_ptr, f_ptr);

%%  ��ȡԤ��
n_outputs = out_len_ptr.Value;
setdatatype(f, 'singlePtr', n_outputs);

%%  �õ��������
Ypre = double(f.Value);
