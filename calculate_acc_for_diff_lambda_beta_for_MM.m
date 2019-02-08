% Discription: calculate the importance order of elements in MM
% Date: 2019/01/27
% Author: Dongdong Ma
function calculate_acc_for_diff_lambda_beta_for_MM()

clc;
tic;
cls_name = {'duokong','guanghua','weizhu'};
base_path = 'E:\电纺丝数据\自己重新整理后\MM_extracted_vec_for_all_cls\';

cls_num = [51, 59, 47];
train_num = [25, 29, 23];

run_num = 20;
threshold_coefficient = 1.2;
first_N = 12;                             % remain the first N elements

diff_lambda_beta_acc_mat = [];
counter1 = 0;
for lambda = 0:0.5:5
    counter1 = counter1 + 1;
    counter2 = 0;
    for beta = 0:0.05:0.5
        counter2 = counter2 + 1;
        acc_vec = [];
        for run = 1:run_num
            for i = 1:length(cls_num)
                % select the train and the test set randomly
                temp = randperm(cls_num(i));
                train_index_set{1,i} = temp(1:train_num(i));
                test_index_set{1,i} = temp(train_num(i)+1:cls_num(i));
            end

            MM_element_num = 16;

            train_vec = [];
            test_vec = [];
            train_label = [];
            test_label = [];

            for ii = 1:length(cls_num)
                current_cls_name = strcat(base_path,cls_name{1,ii},'\',cls_name{1,ii},'_samples.mat');
                temp = load(current_cls_name);
                current_cls_data = temp.all_this_samples;
                miu_mat = zeros(size(current_cls_data,2),MM_element_num-1);
                for j = 1:train_num(ii)
                    jj = train_index_set{1,ii}(j);
                    miu_mat = miu_mat + current_cls_data((jj-1)*MM_element_num+2:jj*MM_element_num,:)';
                end
                miu_mat = miu_mat/train_num(ii);
                miu_mat_set{1,ii} = miu_mat;
            end

            AA = zeros(MM_element_num-1,MM_element_num-1);
            BB = zeros(MM_element_num-1,MM_element_num-1);
            CC = zeros(MM_element_num-1,MM_element_num-1);
            cls_seq = [1,2,3];
            for ii = 1:length(cls_num)
                current_cls_name = strcat(base_path,cls_name{1,ii},'\',cls_name{1,ii},'_samples.mat');
                temp = load(current_cls_name);
                current_cls_data = temp.all_this_samples;
                % caculate AA
                for j = 1:train_num(ii)
                    jj = train_index_set{1,ii}(j);
                    this_G_mat = current_cls_data((jj-1)*MM_element_num+2:jj*MM_element_num,:)';
                    AA = AA + (this_G_mat - miu_mat_set{1,ii})'*(this_G_mat - miu_mat_set{1,ii});
                end
                % caculate BB
                for jj = 1:length(cls_num)
                    BB = BB + (miu_mat_set{1,ii} - miu_mat_set{1,jj})'*(miu_mat_set{1,ii} - miu_mat_set{1,jj});
                end
                % caculate CC
                other_num = cls_seq;
                other_num(ii) = [];
                for j = 1:train_num(ii)
                    jj = train_index_set{1,ii}(j);
                    this_G_mat = current_cls_data((jj-1)*MM_element_num+2:jj*MM_element_num,:)';
                    for kk = 1:length(other_num)
                        this_num = other_num(kk);
                        CC = CC + (this_G_mat - miu_mat_set{1,this_num})'*(this_G_mat - miu_mat_set{1,this_num});
                    end
                end   
            end

            QQ = AA - lambda*BB - beta*CC;
            x0 = rand(size(QQ,1),1);x0 = x0/sum(x0); 
            z0 = rand(size(QQ,1),1);z0 = z0/sum(z0);                     
            y0 = 1;                                                       
            c = zeros(size(x0));
            delta = 0.1; theta = 0.3; epsilon = 10^-5;
            A = [1,ones(1,length(x0)-1)];
            b = 1;

            for ii = 1:1
                [x,y,z,~,F] = self_primal_dual2(QQ,c,A,b,x0,y0,z0,epsilon,theta,delta);
                x0 = x; y0 = y; z0 = z;
            end

            w1 = x0;


            % ********************** update ******************
            for ii = 1:length(cls_num)
                current_cls_name = strcat(base_path,cls_name{1,ii},'\',cls_name{1,ii},'_samples.mat');
                temp = load(current_cls_name);
                current_cls_data = temp.all_this_samples;

                miu_vec = zeros(size(current_cls_data,2),1);
                for j = 1:train_num(ii)
                    jj = train_index_set{1,ii}(j);
                    single_current_cls_data = current_cls_data((jj-1)*MM_element_num+2:jj*MM_element_num,:)';
                    miu_vec = miu_vec + single_current_cls_data*x0;
                end
                miu_vec = miu_vec/train_num(ii);
                miu_vec_set{1,ii} = miu_vec;
            end

            other_center = 1:length(cls_num);
            distance_vec_set = cell(1,length(cls_num));
            for ii = 1:length(cls_num)
                current_cls_name = strcat(base_path,cls_name{1,ii},'\',cls_name{1,ii},'_samples.mat');
                temp = load(current_cls_name);
                current_cls_data = temp.all_this_samples;
                % caculate AA
                distance_vec = zeros(train_num(ii),1);
                for j = 1:train_num(ii)
                    this_other_center = other_center;
                    temp = (other_center == ii);
                    this_other_center(temp) = [];

                    jj = train_index_set{1,ii}(j);
                    this_G_mat = current_cls_data((jj-1)*MM_element_num+2:jj*MM_element_num,:)';
                    this_weighted_G_vec = this_G_mat*x0;
                    
                    min_distance_vec = [];
                    for kk = 1:length(this_other_center)
                        other_index = this_other_center(kk);
                        min_distance_vec = [min_distance_vec,(this_weighted_G_vec - miu_vec_set{1,other_index})'*(this_weighted_G_vec - miu_vec_set{1,other_index})];
                    end
                    min_distance = min(min_distance_vec);

                    distance_vec(j,1) = min_distance - threshold_coefficient*((this_weighted_G_vec - miu_vec_set{1,ii})'*(this_weighted_G_vec - miu_vec_set{1,ii}));
                end
                distance_vec_set{1,ii} = distance_vec;
            end

            hard_num_set = cell(1,length(cls_num));
            threshold_D = 0;
            for ii = 1:length(cls_num)
                distance_vec = distance_vec_set{1,ii};
                hard_num = find(distance_vec < threshold_D);
                hard_num_set{1,ii} = hard_num;
            end

            AA = zeros(MM_element_num-1,MM_element_num-1);
            BB = zeros(MM_element_num-1,MM_element_num-1);
            CC = zeros(MM_element_num-1,MM_element_num-1);
            cls_seq = [1,2,3];
            for ii = 1:length(cls_num)
                current_cls_name = strcat(base_path,cls_name{1,ii},'\',cls_name{1,ii},'_samples.mat');
                temp = load(current_cls_name);
                current_cls_data = temp.all_this_samples;
                % caculate AA
                this_hard_num_vec = hard_num_set{1,ii};
                for jj = 1:length(this_hard_num_vec)
                    temp = this_hard_num_vec(jj);
                    new_jj = train_index_set{1,ii}(temp);
                    this_G_mat = current_cls_data((new_jj-1)*MM_element_num+2:new_jj*MM_element_num,:)';
                    AA = AA + (this_G_mat - miu_mat_set{1,ii})'*(this_G_mat - miu_mat_set{1,ii});
                end
                % caculate BB
                for jj = 1:length(cls_num)
                    BB = BB + (miu_mat_set{1,ii} - miu_mat_set{1,jj})'*(miu_mat_set{1,ii} - miu_mat_set{1,jj});
                end
                % caculate CC
                other_num = cls_seq;
                other_num(ii) = [];
                for jj = 1:length(this_hard_num_vec)
                    temp = this_hard_num_vec(jj);
                    new_jj = train_index_set{1,ii}(temp);
                    this_G_mat = current_cls_data((new_jj-1)*MM_element_num+2:new_jj*MM_element_num,:)';
                    for kk = 1:length(other_num)
                        this_num = other_num(kk);
                        CC = CC + (this_G_mat - miu_mat_set{1,this_num})'*(this_G_mat - miu_mat_set{1,this_num});
                    end
                end   
            end

            QQ = AA - lambda*BB - beta*CC;
            x0 = w1;
            z0 = rand(size(QQ,1),1);z0 = z0/sum(z0);                     
            y0 = 1;
            c = zeros(size(x0));
            A = [1,ones(1,length(x0)-1)];
            b = 1;

            for ii = 1:1
                [x,y,z,XX,F] = self_primal_dual2(QQ,c,A,b,x0,y0,z0,epsilon,theta,delta);
                x0 = x; y0 = y; z0 = z;
            end

            w1 = [0;x0];
            [~,ind] = sort(w1,'descend');
            temp1 = ind(1:first_N);
            temp2 = zeros(size(w1));
            temp2(temp1) = 1;
            w1 = temp2;

            for ii = 1:length(cls_num)
                current_cls_name = strcat(base_path,cls_name{1,ii},'\',cls_name{1,ii},'_samples.mat');
                temp = load(current_cls_name);
                current_cls_data = temp.all_this_samples;

                transfered_vec = feat_transfer3(current_cls_data,cls_num(ii),w1',MM_element_num);            % 主函数

                train_vec = [train_vec;transfered_vec(train_index_set{1,ii},:)];
                train_label = [train_label,ones(1,train_num(ii))*ii];
                test_vec = [test_vec;transfered_vec(test_index_set{1,ii},:)];
                test_label = [test_label,ones(1,(cls_num(ii)-train_num(ii)))*ii];
            end
            % classification using the 1-NN classifier
            trainClassIDs = train_label;
            testClassIDs = test_label;
            trainNum = size(train_vec,1);
            testNum = size(test_vec,1);
            DM = zeros(testNum,trainNum);
            for i=1:testNum
                test = test_vec(i,:);        
                DM(i,:) = distMATChiSquare(train_vec,test)';
            end
            [acc1,~] = ClassifyOnNN(DM,trainClassIDs,testClassIDs);
            acc_vec = [acc_vec,acc1];
        end
        diff_lambda_beta_acc_mat(counter1,counter2) = mean(acc_vec);
    end
end

toc


