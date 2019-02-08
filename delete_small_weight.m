% delete the elements with small weights

function new_w = delete_small_weight(w,delete_num)

new_w = w;
[sorted_w,ind_vec] = sort(w);

sorted_w(1:delete_num) = 0;
sorted_w = sorted_w/sum(sorted_w);
new_w(ind_vec) = sorted_w;
