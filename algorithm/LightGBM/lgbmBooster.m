classdef lgbmBooster < handle
    properties
        pointer
    end
   
    methods
        function obj = lgbmBooster(datasetFileOrDef, params)
            if ischar(datasetFileOrDef)
                assert(nargin == 1)

                if ~contains(datasetFileOrDef, newline)
                    file = datasetFileOrDef;
                    [ret, ~, ~, booster] = calllib('lib_lightgbm', 'LGBM_BoosterCreateFromModelfile', file, 0, libpointer('voidPtr'));
                    checkError(ret)
                    obj.pointer = booster;
                else
                    definition = datasetFileOrDef;
                    [ret, ~, ~, booster] = calllib('lib_lightgbm', 'LGBM_BoosterLoadModelFromString', definition, 0, libpointer('voidPtr'));
                    checkError(ret)
                    obj.pointer = booster;
                end

            else

                if nargin == 1
                    params = '';
                end

                if isa(params,'containers.Map')
                    params = parametersString(params);
                end

                dataset = datasetFileOrDef;
                assert(isa(dataset, 'lgbmDataset'))
                checkPointer(dataset)
                [ret, ~, ~, booster] = calllib('lib_lightgbm', 'LGBM_BoosterCreate', dataset.pointer, params, libpointer('voidPtr'));
                checkError(ret)
                obj.pointer = booster;
            end
        end
        
        function free(obj)
            checkPointer(obj)
            ret = calllib('lib_lightgbm', 'LGBM_BoosterFree', obj.pointer);
            checkError(ret)
            obj.pointer = [];
        end        
                 
        function delete(obj)            
            free(obj)
        end
        
        function merge(obj, obj2)
            checkPointer(obj)
            assert(isa(obj2, 'lgbmBooster'))
            checkPointer(obj2)
            ret = calllib('lib_lightgbm', 'LGBM_BoosterMerge', obj.pointer, obj.pointer2);
            checkError(ret)
        end

        function addValidationData(obj, dataset)
            checkPointer(obj)
            assert(isa(dataset, 'lgbmDataset'))
            checkPointer(dataset)
            ret=calllib('lib_lightgbm', 'LGBM_BoosterAddValidData', obj.pointer, dataset.pointer);
            checkError(ret)
        end
        
        function resetTrainingData(obj, dataset)            
            checkPointer(obj)
            assert(isa(dataset, 'lgbmDataset'))
            checkPointer(dataset)
            ret = calllib('lib_lightgbm', 'LGBM_BoosterResetTrainingData', obj.pointer, dataset.pointer);
            checkError(ret)
        end
        
        function resetParameters(obj, params)
            checkPointer(obj)
            ret = calllib('lib_lightgbm', 'LGBM_BoosterResetParameter', obj.pointer,params);
            checkError(ret)
        end
        
        function nclasses = numClasses(obj)
            checkPointer(obj)
            [ret, ~, nclasses] = calllib('lib_lightgbm', 'LGBM_BoosterGetNumClasses', obj.pointer, 0);
            checkError(ret)
        end
        
        function nfeatures = numFeatures(obj)
            checkPointer(obj)
            [ret, ~, nfeatures] = calllib('lib_lightgbm', 'LGBM_BoosterGetNumFeature', obj.pointer, 0);
            checkError(ret)
        end
        
        function names = featureNames(obj)
            checkPointer(obj)
            maxlen = 100;
            n = numFeatures(obj);
            names = cell(1, n);

            for i = 1: length(names)
                names{i} = repmat(' ', 1, maxlen);
            end

            [ret, ~, len, names] = calllib('lib_lightgbm', 'LGBM_BoosterGetFeatureNames', obj.pointer, 0, libpointer('stringPtrPtr', names));
            checkError(ret)
            assert(len == n)
            names = names';
        end
        
        function value = getLeaveValue(obj, tree, leaf)
            checkPointer(obj)
            [ret, ~, value] = calllib('lib_lightgbm', 'LGBM_BoosterGetLeafValue', obj.pointer, tree, leaf, 0);
            checkError(ret)
        end
        
        function setLeaveValue(obj, tree, leaf, value)
            checkPointer(obj)
            ret = calllib('lib_lightgbm', 'LGBM_BoosterSetLeafValue', obj.pointer, tree, leaf, value);
            checkError(ret)
        end
        
        function iteration = getIteration(obj)
            checkPointer(obj)
            [ret, ~, iteration] = calllib('lib_lightgbm', 'LGBM_BoosterGetCurrentIteration', obj.pointer, 0);
            checkError(ret)
        end

        function finished = updateOneIter(obj)
            checkPointer(obj)
            [ret, ~, finished] = calllib('lib_lightgbm', 'LGBM_BoosterUpdateOneIter', obj.pointer, 0);
            checkError(ret)
        end
        
        function rollbackOneIter(obj)
            checkPointer(obj)
            ret = calllib('lib_lightgbm', 'LGBM_BoosterRollbackOneIter', obj.pointer);
            checkError(ret)
        end
        
        function count = evalCounts(obj)
            checkPointer(obj)
            [ret, ~, count] = calllib('lib_lightgbm', 'LGBM_BoosterGetEvalCounts', obj.pointer, 0);
            checkError(ret)
        end
        
        function names = evalNames(obj)
            checkPointer(obj)
            maxlen = 100;
            n = evalCounts(obj);
            names = cell(1, n);

            for i = 1: length(names)
                names{i} = repmat(' ', 1, maxlen);
            end

            [ret, ~, len, names] = calllib('lib_lightgbm', 'LGBM_BoosterGetEvalNames', obj.pointer, 0, libpointer('stringPtrPtr', names));
            checkError(ret)
            assert(len == n);
        end
       
        function eval = getEval(obj, idx)
            checkPointer(obj)

            if nargin == 1
                idx = 0;
            end

            n = evalCounts(obj);
            [ret, ~, len, eval] = calllib('lib_lightgbm', 'LGBM_BoosterGetEval', obj.pointer, idx, 0, libpointer('doublePtr', zeros(1, n)));
            checkError(ret)
            assert(len == n)
        end

        function predictions = predict(obj, idx)
            checkPointer(obj)

            if nargin == 1
                idx = 0;
            end

            [ret, ~, npredict] = calllib('lib_lightgbm', 'LGBM_BoosterGetNumPredict', obj.pointer, idx, 0);
            checkError(ret);
            npredict = double(npredict);
            nclasses = double(numClasses(obj));

            [ret, ~, len, predictions] = calllib('lib_lightgbm', 'LGBM_BoosterGetPredict', obj.pointer, idx, 0, libpointer('doublePtr', zeros(1, npredict)));
            checkError(ret)
            assert(len == npredict)
            predictions = reshape(predictions, npredict / nclasses, nclasses);

        end

        function predictFile(obj, input, output, iteration, type, inputHasHeader, params)
            checkPointer(obj)

            if nargin < 4
                iteration = 0;
            end

            if nargin < 5
                type = 'normal';
            end

            if nargin < 6
                inputHasHeader = 0;
            end

            if nargin < 7
                params = '';
            end

            if isa(params, 'containers.Map')
                params = parametersString(params);
            end

            if strcmp(type, 'normal')
                type = 0;
            elseif strcmp(type, 'raw_score') || strcmp(type, 'raw') 
                type = 1;
            elseif strcmp(type, 'leaf_index') || strcmp(type, 'leaf')
                type = 2;
            else
                error('type should be one of: normal raw_score (or raw) leaf_index (or leaf)')
            end

            ret = calllib('lib_lightgbm', 'LGBM_BoosterPredictForFile', obj.pointer, input, inputHasHeader, type, iteration, params, output);
            checkError(ret)
        end
        
        function predictions = predictMatrix(obj, matrix, iteration, type, dataType, params)
            checkPointer(obj)
            
            if nargin < 3
                iteration = 0;
            end

            if nargin < 4
                type = 'normal';
            end
            
            if nargin < 5
                dataType = 'float';
            end

            if nargin < 6
                params = '';
            end

            if isa(params, 'containers.Map')
                params = parametersString(params);
            end

            if strcmp(dataType, 'float')
                dataType = 0;
                ptrType = 'singlePtr';
            elseif strcmp(dataType, 'double')
                dataType = 1;
                ptrType = 'doublePtr';
            else
                error('dataType sould be float or double')
            end

            if strcmp(type, 'normal')
                type = 0;
            elseif strcmp(type, 'raw_score') || strcmp(type, 'raw') 
                type = 1;
            elseif strcmp(type, 'leaf_index') || strcmp(type, 'leaf')
                type = 2;
            else
                error('type should be one of: normal raw_score (or raw) leaf_index (or leaf)')
            end

            nrow = size(matrix, 1);
            ncol = size(matrix, 2);
            isrowmajor = 0;
            data = reshape(matrix, 1, numel(matrix));
            data = libpointer(ptrType, data);

            [ret, ~, npred] = calllib('lib_lightgbm', 'LGBM_BoosterCalcNumPredict', obj.pointer, nrow, type, iteration, 0);
            checkError(ret)
            npred = double(npred);
            
            [ret, ~, ~, ~, len, predictions]=calllib('lib_lightgbm', 'LGBM_BoosterPredictForMat', obj.pointer, data, dataType, nrow, ncol, isrowmajor, type, iteration, params, 0, libpointer('doublePtr', zeros(1, npred)));
            checkError(ret)
            assert(len == npred)
            predictions = reshape(predictions, npred / nrow, nrow)';
        end

        function output = modelToString(obj, iteration)
            checkPointer(obj)

            if nargin == 1
                iteration = 0;
            end
           
            [ret, ~, outLen] = calllib('lib_lightgbm', 'LGBM_BoosterSaveModelToString', obj.pointer, iteration, 0, 0, '');
            checkError(ret)
            [ret, ~, ~, output] = calllib('lib_lightgbm', 'LGBM_BoosterSaveModelToString', obj.pointer, iteration, outLen, 0, repmat(' ', 1, outLen));
            checkError(ret)
        end

        function output = dumpModel(obj, iteration)
            checkPointer(obj)

            if nargin == 1
                iteration = 0;
            end

            [ret, ~, outLen] = calllib('lib_lightgbm', 'LGBM_BoosterDumpModel', obj.pointer, iteration, 0, 0, '');
            checkError(ret)

            [ret, ~, ~, output] = calllib('lib_lightgbm', 'LGBM_BoosterDumpModel', obj.pointer, iteration, outLen, 0, repmat(' ', 1, outLen));
            checkError(ret)
        end
        
        function saveModel(obj, file, iteration)
            checkPointer(obj)

            if nargin == 2
                iteration = 0;
            end

            ret = calllib('lib_lightgbm', 'LGBM_BoosterSaveModel', obj.pointer, iteration, file);
            checkError(ret)
        end
        
        function res = importance(obj, type, iteration)

            if nargin < 3
                iteration = 0;
            end

            trees = jsondecode(obj.dumpModel(iteration));
            res = zeros(1, 1 + trees.max_feature_idx);

            for i = 1: length(trees.tree_info)
                res = processTree(trees.tree_info(i).tree_structure, type, res);
            end

        end
    end
end

function checkPointer(obj)
    if ~isa(obj.pointer, 'lib.pointer')
        error('the object is not associated to a pointer')
    end
end

function checkError(ret)
    if ret ~= 0
        error(calllib('lib_lightgbm', 'LGBM_GetLastError'))
    end
end

function str = parametersString(map)
    keys = map.keys;
    values = map.values;
    for i = 1: map.length
        if i == 1
            str = '';
        else
            str = [str ' '];
        end
        str = [str keys{i} '=' num2str(values{i})];
    end
end

function res = processTree(tree, type, res)
    if any(cellfun(@(x) strcmp(x, 'split_feature'), fields(tree)))
        if tree.split_gain > 0
            i = tree.split_feature + 1;
            if strcmp(type, 'gain')
                res(i) = res(i) + tree.split_gain;
            elseif strcmp(type, 'split')
                res(i) = res(i) + 1;
            else
                error('unrecognized importance type (should be gain or split)')
            end
        end
        res = processTree(tree.left_child , type, res);
        res = processTree(tree.right_child, type, res);
    end
end