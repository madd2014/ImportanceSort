% feature concatenation function
% Date: 2019/01/10
% Author: Dongdong Ma
function transfered_vec = feat_transfer3(current_cls_data,cls_number,coefficients,each_cls_samples)

abs_extract_set = find(coefficients>0);
for ii = 1:cls_number
    extract_num_set = current_cls_data((ii-1)*each_cls_samples + abs_extract_set,:);
    transfered_vec(ii,:) = reshape(extract_num_set',1,length(extract_num_set(:)));
end









