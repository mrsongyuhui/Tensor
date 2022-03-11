clc;
clear all;
warning('on'); %should be the best one
%% System parameters
Mx = 8; % Antenna number x-axis
Mz = 8; % Antenna number z-axis
N = 20; % Subcarrier number, i.e., SCS=15KHz
SNR = 5;
fc = 28e9; % Carrier frequency
fs = N*240e3; % Bandwidth
Td = N/fs; % Data length = 1/SCS
c = 3e8; % Light speed 
lambda_c = c/fc; % Carrier wavelength
d = lambda_c/2; % Antenna spacing
P = 2; % Path number
L = 3; % Dictionary numbers
R = 5; % Coefficient number
T = 30; % Iteration times
Experiment_setting = {Mx, Mz, N, SNR, fc, fs, Td, c, lambda_c, d, P, L, R, T};
%% Simulation result
SNR_list = -10:5:10;
parfor i = 1:length(SNR_list)
    Experiment_setting = {Mx, Mz, N, SNR, fc, fs, Td, c, lambda_c, d, P, L, R, T};
    Experiment_setting(4) = num2cell(SNR_list(i));
    for j = 1:200
        % Generate parameters randomly
        alpha = randn([1,P]) + 1i*randn([1,P]);ones(1,P);
        w_theta_temp = linspace(-pi,pi,100*Mx);
        w_theta_temp_index = randi(10+P)*Mx;
        w_theta = w_theta_temp(linspace(w_theta_temp_index,w_theta_temp_index+(P-1)*Mx,P));[2.2,-1.3];
        w_varphi_temp = linspace(-pi,pi,100*Mz);
        w_varphi_temp_index = randi(10+P)*Mz;
        w_varphi = w_varphi_temp(linspace(w_varphi_temp_index,w_varphi_temp_index+(P-1)*Mz,P));[-2.3,0.85];(rand([1,P])-0.5)*2*pi;
        near = 10;10*(Mx-1)*lambda_c/2;
        far = 100;100*(Mx-1)*lambda_c/2;
        tau = ((far-near)*rand([1,P])+near)/c;
        w_etatau_temp = linspace(near,far,100*N)/c/Td*2*pi;
        w_etatau_temp_index = randi(10+P)*N;
        w_etatau = w_etatau_temp(linspace(w_etatau_temp_index,w_etatau_temp_index+(P-1)*N,P));[2.5,0.2];(tau/Td)*2*pi;(rand([1,P])-0.5)*2*pi;
        % Channel estimation
        path_setting = {alpha, w_theta, w_varphi, w_etatau};
        %[ARDCPD_NMSE(j,i)] = ARDCPD_result(Experiment_setting, path_setting);
        %[zhaoqibin_NMSE(j,i)] = zhaoqibin_result(Experiment_setting, path_setting);
        %[HOSVD_NMSE(j,i)] = hosvd_result(Experiment_setting, path_setting);
        [CP_FULL_original_SBL_DL_share_NMSE(j,i)]=CP_FULL_original_SBL_DL_share(Experiment_setting, path_setting);
        %[Intra_Correlation_CP_SBL_share_NMSE(j,i)]=Intra_Correlation_CP_SBL_share(Experiment_setting, path_setting);
        %[CPD_NMSE(j,i)] = cpd_result(Experiment_setting, path_setting);
        [Intra_Correlation_Tucker_SBL_NMSE(j,i)] = Intra_Correlation_Tucker_SBL(Experiment_setting, path_setting);
        [Intra_Correlation_CP_SBL_NMSE(j,i)] = Intra_Correlation_CP_SBL(Experiment_setting, path_setting);
%         [Traditional_FULL_SBL_DL_NMSE(j,i)] = Traditional_FULL_SBL_DL(Experiment_setting, path_setting);
%         [CP_FULL_SBL_DL_NMSE(j,i)] = CP_FULL_SBL_DL(Experiment_setting, path_setting);
%         [CP_EM_SBL_DL_NMSE(j,i)] = CP_EM_SBL_DL(Experiment_setting, path_setting);
        %[CP_FULL_original_SBL_DL_NMSE(j,i)] = CP_FULL_original_SBL_DL(Experiment_setting, path_setting);
%         [CP_FULL_original_SBL_DL_Mixed_NMSE(j,i)] = CP_FULL_original_SBL_DL_Mixed(Experiment_setting, path_setting);

%         [CP_FULL_original_SBL_DL_Orthogonal_NMSE(j,i)] = CP_FULL_original_SBL_DL_Orthogonal(Experiment_setting, path_setting);
%         [CP_EM_original_SBL_DL_NMSE(j,i)] = CP_EM_original_SBL_DL(Experiment_setting, path_setting);
        [Tucker_SBL_DL_JSTSP_NMSE(j,i)] = Tucker_SBL_DL_JSTSP(Experiment_setting, path_setting);
%         CP_FULL_SBL_SR_NMSE(j,i) = CP_FULL_SBL_SR(Experiment_setting, path_setting);
%         Tucker_SBL_SR_JSTSP_NMSE(j,i) = Tucker_SBL_SR_JSTSP(Experiment_setting, path_setting);
%         Parameterized_SBL_NMSE(j,i) = Parameterized_SBL(Experiment_setting, path_setting);
%         Truncated_Parameterized_SBL_NMSE(j,i) = Truncated_Parameterized_SBL(Experiment_setting, path_setting);
    end
end

figure(1);
MarkerSize = 6;
LineWidth = 1.2;
semilogy(SNR_list,(mean(CP_FULL_original_SBL_DL_share_NMSE)),'o--','color','#EDB120','MarkerSize',MarkerSize,'LineWidth',LineWidth);
hold on;
semilogy(SNR_list,(mean(Intra_Correlation_CP_SBL_NMSE)),'*-','color','#77AC30','MarkerSize',MarkerSize,'LineWidth',LineWidth);
semilogy(SNR_list,(mean(Tucker_SBL_DL_JSTSP_NMSE)),'o--','color','#0072BD','MarkerSize',MarkerSize,'LineWidth',LineWidth);
semilogy(SNR_list,(mean(Intra_Correlation_Tucker_SBL_NMSE)),'*-','color','#7E2F8E','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% semilogy(SNR_list,mean(Traditional_FULL_SBL_DL_NMSE),'o-','color','#77AC30','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% semilogy(SNR_list,mean(CP_FULL_SBL_DL_NMSE),'o-','color','#7E2F8E','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% semilogy(SNR_list,mean(CP_EM_SBL_DL_NMSE),'o-','color','#EDB120','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% semilogy(SNR_list,mean(CP_FULL_original_SBL_DL_Mixed_NMSE),'d-','color','#EDB120','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% semilogy(SNR_list,mean(CP_EM_original_SBL_DL_NMSE),'s--','color','#EDB120','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% semilogy(SNR_list,mean(CP_FULL_SBL_SR_NMSE),'o-','color','#D95319','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% semilogy(SNR_list,mean(Tucker_SBL_SR_JSTSP_NMSE),'o-','color','#4dBEEE','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% semilogy(SNR_list,mean(Parameterized_SBL_NMSE),'s-','color','#A2142F','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% semilogy(SNR_list,mean(Truncated_Parameterized_SBL_NMSE),'o-','color','#A2142F','MarkerSize',MarkerSize,'LineWidth',LineWidth);
%semilogy(SNR_list,(mean(HOSVD_NMSE)),'^-','color','#A2142F','MarkerSize',MarkerSize,'LineWidth',LineWidth);
%semilogy(SNR_list,(mean(CPD_NMSE)),'^--','color','#4dBEEE','MarkerSize',MarkerSize,'LineWidth',LineWidth);
%semilogy(SNR_list,(mean(CP_FULL_original_SBL_DL_share_NMSE)),'>--','color','#D95319','MarkerSize',MarkerSize,'LineWidth',LineWidth);
%semilogy(SNR_list,(mean(zhaoqibin_NMSE)),'>-','color','#4dBEEE','MarkerSize',MarkerSize,'LineWidth',LineWidth);
%semilogy(SNR_list,(mean(ARDCPD_NMSE)),'<-','color','#4dBEEE','MarkerSize',MarkerSize,'LineWidth',LineWidth);
xlabel('SNR(dB)','interpreter','latex', 'FontSize', 20, 'FontName', 'Times New Roman');
ylabel('NMSE','interpreter','latex', 'FontSize', 20, 'FontName', 'Times New Roman');
% axis([0,max(carrier_numbers), low*min(whole_channel_axis), high*max(whole_channel_axis)]);
legend('Original-CP','Correlated-CP','Original-Tucker','Correlated-Tucker','interpreter','latex','FontSize', 20, 'FontName', 'Times New Roman');

%% Channel estimation using CP based full original SBL DL_share
function [normalized_error] = CP_FULL_original_SBL_DL_share(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
% Parameter Initialization
D_size = {Mx,Mz,N};
for l = 1:L
    D_shape_prior{l} = ones([R,1])*eps;
    D_rate_prior{l} = ones([R,1])*eps;
    D_precision{l} = diag(ones([R,1]));
    temp = randn([D_size{l},R]) + 1i*randn([D_size{l},R]);
    D_mean{l} = temp./abs(temp);
end
x_shape_prior = ones([R,1])*eps;
x_rate_prior = ones([R,1])*eps;
x_precision = diag(ones([R,1]));
x_mean = randn([R,1]) + 1i*randn([R,1]);
n_shape_prior = eps;
n_rate_prior = eps;
% Initialization
for l = 1:L
    D_shape{l} = ones([R,1])*eps;
    D_rate{l} = ones([R,1])*eps;
end
x_shape = ones([R,1])*eps;
x_rate = ones([R,1])*eps;
n_shape = 1;
n_rate = eps;

for t = 1:T
    
    % update D
    for l = 1:L
        % calculate D_mean_square_exclusive
        D_mean_square_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                D_mean_square_exclusive = D_mean_square_exclusive.*(D_size{i}*inv(D_precision{i})+D_mean{i}'*D_mean{i});
            end
        end
        % calculate D_precision
        temp = diag(kron(conj(x_mean),x_mean)+reshape(inv(x_precision),[],1));
        D_precision{l} = n_shape/n_rate*conj(D_mean_square_exclusive)+diag(D_shape{l}./D_rate{l});
        % calculate D_mean_exclusive
        D_mean_exclusive = ones(1,R);
        for i = L:-1:1
            if i ~= l
                D_mean_exclusive = khatrirao(D_mean_exclusive,conj(D_mean{i}));
            end
        end
        % calculate D_mean
        D_mean{l} = n_shape/n_rate*double(tenmat(noisy_channel,l))*D_mean_exclusive*inv(D_precision{l});
        % calculate D_shape and D_rate
        D_shape{l} = D_size{l}+D_shape_prior{l};
        D_rate{l} = diag(D_size{l}*inv(D_precision{l})+D_mean{l}'*D_mean{l})+D_rate_prior{l};
    end
    
%     temp=D_prior_seperate_precision_temp{1}+D_prior_seperate_precision_temp{2}+D_prior_seperate_precision_temp{3}+...
%      x_prior_precision.*diag(inv(x_precision)+x_mean*x_mean');
 
     temp0 = 0;
     temp1 = 0;x_shape./x_rate.*diag(inv(x_precision)+x_mean*x_mean');
     for l=1:L
         temp0 = temp0 + D_size{l};
         temp1 = temp1 + diag(D_size{l}*inv(D_precision{l})+D_mean{l}'*D_mean{l});
     end
     
     for l=1:L
         D_shape{l} = temp0 + D_shape_prior{l};
         D_rate{l} = temp1 + D_rate_prior{l};
     end 
    
    % updata x
    D_mean_square = 1;
    for l = L:-1:1
        D_mean_square = D_mean_square.*(D_size{l}*inv(D_precision{l})+D_mean{l}'*D_mean{l});
    end
    % calculate x_precision
    x_precision = n_shape/n_rate*D_mean_square+diag(x_shape./x_rate);
    D_mean_all = ones(1,R);
    for l = L:-1:1
        D_mean_all = khatrirao(D_mean_all,D_mean{l});
    end
    % calculate x_mean
    x_mean = ones(R,1);n_shape/n_rate*inv(x_precision)*D_mean_all'*vectorized_noisy_channel;
    % calculate x_shape annd x_rate
    x_shape = 1+x_shape_prior;
    x_mean_square = inv(x_precision)+x_mean*x_mean';
    x_rate = diag(x_mean_square).*D_shape{1}./D_rate{1}+x_rate_prior;
    
%     x_prior_precision = 1./diag(inv(x_precision)+x_mean*x_mean')./...
%         D_prior_seperate_precision{1};
    
    % update n
    n_shape = prod(cell2mat(D_size))+n_shape_prior;
    n_rate = vectorized_noisy_channel'*vectorized_noisy_channel ...
        - ones(R,1)'*D_mean_all'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_mean_all*ones(R,1) ...
        + ones(R,1)'*D_mean_square*ones(R,1) + n_rate_prior;
    
end
normalized_error = sum(abs(D_mean_all*ones(R,1)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');

% sum(abs(reshape(double(hosvd(tensor(noisy_channel),2,'rank',[2,2,2])),[],1)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all')
% [U,~]=cpd((noisy_channel),2);
% sum(abs(sum(khatrirao(U{3},U{2},U{1}),2)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all')

end


%% Channel estimation using intra-correlation CP SBL
function [normalized_error] = Intra_Correlation_CP_SBL_share(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
% Parameter Initialization
D_precision = {}; D_mean = {}; D_shape = {}; D_rate = {};
D_size = {Mx,Mz,N};
for l = 1:L
    D_prior_seperate_precision{l} = ones([R,1]);
    temp = rand([D_size{l},D_size{l}]);D_prior_common_precision{l} = (temp*temp');eye(D_size{l});
    temp = rand([D_size{l}*R,D_size{l}*R]);D_precision{l} = (temp*temp');eye(D_size{l}*R);
    temp = randn([D_size{l}*R,1]) + 1i*randn([D_size{l}*R,1]);
    D_mean{l} = temp./abs(temp);
end
x_prior_precision = ones([R,1]);
x_precision = diag(ones([R,1]));
x_mean = randn([R,1]) + 1i*randn([R,1]);
n_precision = 1/eps;


for t = 1:T
    % update D
    for l = 1:L
        % calculate D_mean_square_exclusive
        D_mean_square_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                D_covatiance{i} = cellfun(@sum,cellfun(@diag,mat2cell(inv(D_precision{i}).',ones(R,1)*D_size{i}, ones(R,1)*D_size{i}),'UniformOutput',false));
                D_mean_square_exclusive = D_mean_square_exclusive.*(D_covatiance{i}+reshape(D_mean{i},D_size{i},R)'*reshape(D_mean{i},D_size{i},R));
            end
        end
        % calculate D_precision
        temp = diag(kron(x_mean,conj(x_mean))+reshape(inv(x_precision).',[],1));
        D_precision{l} = n_precision*kron(reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l}))+kron(diag(D_prior_seperate_precision{l}),D_prior_common_precision{l});
        % calculate D_mean_exclusive
        D_mean_exclusive = ones(1,R);
        for i = L:-1:1
            if i ~= l
                D_mean_exclusive = khatrirao(D_mean_exclusive,reshape(D_mean{i},D_size{i},R));
            end
        end
        % Reduce complexity for inv(D_precision{l})
%         D = kron(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l}));
%         C = kron(diag(D_prior_seperate_precision{l}),D_prior_common_precision{l});
        
%         inv_D_precision{l}=inv(eye(R*D_size{l})+inv(D)*C)*inv(D);

%         inv_D_precision{l}=inv(eye(R*D_size{l})+inv(kron(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l})))*kron(diag(D_prior_seperate_precision{l}),D_prior_common_precision{l})...
%             )*inv(kron(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l})));
%         
%           % Can work
%         inv_D_precision{l}=inv(eye(R*D_size{l})+kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R))*...
%             diag(D_prior_seperate_precision{l}),...
%             D_prior_common_precision{l})...
%             )*inv(kron(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l})));
%         
        


        [Eigenmatrix_P, Eigenvalue_P] = eig(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R))*diag(D_prior_seperate_precision{l}));
%         norm(Eigenmatrix_P*Eigenvalue_P*Eigenmatrix_P'-buzhidao,'fro');
        [Eigenmatrix_Q, Eigenvalue_Q] = eig(D_prior_common_precision{l});
        
%         inv_D_precision{l}=inv(kron(Eigenmatrix_P*Eigenmatrix_P',Eigenmatrix_Q*Eigenmatrix_Q')+...
%             kron(Eigenmatrix_P*Eigenvalue_P*Eigenmatrix_P',Eigenmatrix_Q*Eigenvalue_Q*Eigenmatrix_Q')...
%             )*kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));
%         

%           inv_D_precision{l}=inv(kron(Eigenmatrix_P,Eigenmatrix_Q)*kron(Eigenmatrix_P,Eigenmatrix_Q)'+...
%               kron(Eigenmatrix_P,Eigenmatrix_Q)*kron(Eigenvalue_P,Eigenvalue_Q)*kron(Eigenmatrix_P,Eigenmatrix_Q)'...
%             )*kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));
%        
%          % Can work
%         inv_D_precision{l}=kron(Eigenmatrix_P,Eigenmatrix_Q)*inv(eye(R*D_size{l})+...
%               kron(Eigenvalue_P,Eigenvalue_Q)...
%             )*kron(inv(Eigenmatrix_P),inv(Eigenmatrix_Q))...
%             *kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));

        
        inv_D_precision{l}=kron(Eigenmatrix_P,Eigenmatrix_Q)*diag(...
            1./(1+diag(kron(Eigenvalue_P,Eigenvalue_Q)))...
            )*kron(inv(Eigenmatrix_P),inv(Eigenmatrix_Q))...
            *kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));

        
        
        
        
%         inv_D_precision{l} = inv(eye(R*D_size{l}) + kron((inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R))*diag(D_prior_seperate_precision{l}))...
%        ,D_prior_common_precision{l}));
%         
%         inv_D_precision{l} = kron(Eigenmatrix_P,Eigenmatrix_Q)' * diag(1./(1+diag(kron(Eigenvalue_P,Eigenvalue_Q)))) * kron(Eigenmatrix_P,Eigenmatrix_Q) ...
%             * kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));
%         sum(abs(inv_D_precision{l}-inv(D_precision{l})),'all') / sum(abs(inv(D_precision{l})),'all');
        
        % calculate D_mean
        D_mean{l} = n_precision*inv_D_precision{l}*kron(D_mean_exclusive*diag(x_mean),eye(D_size{l}))'*reshape(double(tenmat(noisy_channel,l)),[],1);
    end
    
% %     update x % Transpose of D_mean_square 
%     D_mean_square_T = 1;
%     for l = L:-1:1
%         D_mean_square_T = D_mean_square_T.*cellfun(@sum,cellfun(@diag,mat2cell(inv(D_precision{l})+D_mean{l}*D_mean{l}',ones(R,1)*D_size{l}, ones(R,1)*D_size{l}),'UniformOutput',false));
%     end
    % update x
    D_mean_square = 1;
    for l = L:-1:1
        D_covatiance{l} = cellfun(@sum,cellfun(@diag,mat2cell(inv_D_precision{l}.',ones(R,1)*D_size{l}, ones(R,1)*D_size{l}),'UniformOutput',false));
        D_mean_square = D_mean_square.*(D_covatiance{l}+reshape(D_mean{l},D_size{l},R)'*reshape(D_mean{l},D_size{l},R));
    end
    
    % calculate x_precision
    x_precision = n_precision*D_mean_square+diag(x_prior_precision.*D_prior_seperate_precision{1});
    D_mean_all = ones(1,R);
    for l = L:-1:1
        D_mean_all = khatrirao(D_mean_all,reshape(D_mean{l},D_size{l},R));
    end
    % calculate x_mean
    x_mean = n_precision*inv(x_precision)*D_mean_all'*vectorized_noisy_channel;
    
    % calculate x_prior_precision
    x_prior_precision = 1./diag(inv(x_precision)+x_mean*x_mean')./...
        D_prior_seperate_precision{1};
    
    % update n_precision
%         D_mean_square = 1;
%     for l = L:-1:1
%         D_covatiance{l} = cellfun(@sum,cellfun(@diag,mat2cell(inv_D_precision{l}.',ones(R,1)*D_size{l}, ones(R,1)*D_size{l}),'UniformOutput',false));
%         D_mean_square = D_mean_square.*(D_covatiance{l}+reshape(D_mean{l},D_size{l},R)'*reshape(D_mean{l},D_size{l},R));
%     end 1/noise_variance;
    
    n_precision = prod(cell2mat(D_size)) / ...
        (vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_mean_all'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_mean_all*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(D_mean_square,[],1));
    
%     n_precision=prod(cell2mat(D_size)) /((vectorized_noisy_channel-D_mean_all*x_mean)'*(vectorized_noisy_channel-D_mean_all*x_mean)+0);trace(D_mean_all'*D_mean_all*inv(x_precision))
%     n_precision = 1/noise_variance;
    
%     n_precision = prod(cell2mat(D_size))  / ((vectorized_noisy_channel-D_mean_all*x_mean)'*(vectorized_noisy_channel-D_mean_all*x_mean));
    % D_prior_seperate_precision
    for l=1:L
        D_prior_seperate_precision_temp{l} = diag(cellfun(@trace,cellfun(@(x)D_prior_common_precision{l}*x,mat2cell(inv(D_precision{l})+D_mean{l}*D_mean{l}',ones(R,1)*D_size{l}, ones(R,1)*D_size{l}),'UniformOutput',false)));
%         D_prior_seperate_precision{l} = D_size{l}./D_prior_seperate_precision_temp;
%         temp(temp<0) = 0;
%         D_prior_seperate_precision{l} = temp;
    end
    
    temp=D_prior_seperate_precision_temp{1}+D_prior_seperate_precision_temp{2}+D_prior_seperate_precision_temp{3}+...
        x_prior_precision.*diag(inv(x_precision)+x_mean*x_mean');
    for l=1:L
        D_prior_seperate_precision{l} = (D_size{1}+D_size{2}+D_size{3}+1)./temp;
    end
    
%     % D_prior_common_precision
%     for l=1:L
%         temp_1 = kron(reshape(D_mean{l},D_size{l},R),conj(reshape(D_mean{l},D_size{l},R)));
%         temp_2 = reshape(diag(D_prior_seperate_precision{l}),[],1);
%         D_prior_common_precision{l} = inv(reshape(temp_1*temp_2,D_size{l},D_size{l}).'/R);
%     end
        % D_prior_common_precision
    for l=1:L
        D_prior_common_precision_temp = mat2cell(inv_D_precision{l}+D_mean{l}*D_mean{l}',ones(R,1)*D_size{l}, ones(R,1)*D_size{l});
        D_prior_common_precision_temp_temp = cellfun(@(a,b)a*b,num2cell(D_prior_seperate_precision{l}),D_prior_common_precision_temp(logical(eye(R,R))),'UniformOutput',false);
        D_prior_common_precision{l} = pinv(sum(cat(3,D_prior_common_precision_temp_temp{:}),3).'/R);
        D_prior_common_precision{l} = eye(D_size{l});D_prior_common_precision{l}/norm(D_prior_common_precision{l},'fro');
%      temp = ones(D_size{L})*0.9; temp(logical((eye(D_size{L})))) = ones(D_size{L},1);
%         D_prior_common_precision{L} = temp;           
        
%         D_prior_seperate_precision{l} = D_prior_seperate_precision{l}*norm(D_prior_common_precision{l},'fro');
    end

end
% 1/n_precision > noise_variance;
% sum(abs(reshape(double(hosvd(tensor(noisy_channel),2,'rank',[2,2,2])),[],1)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all')

normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');

end

%% Channel estimation using ARDCPD
function [normalized_error] = ARDCPD_result(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, SNR, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});

% 
% lena512 = imread('lena512color.tiff');
% lena512 = imresize(lena512,0.125);
% lena512 = double(lena512);
% imshow(uint8(lena512(:,:,2)))


R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[original_channel, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
% Parameter Initialization
D_precision = {}; D_mean = {}; D_shape = {}; D_rate = {};
D_size = {Mx,Mz,N};
for l = 1:L
    D_prior_seperate_precision{l} = ones([R,1]);
    temp = rand([D_size{l},D_size{l}]);D_prior_common_precision{l} = (temp*temp');eye(D_size{l});
    temp = rand([D_size{l}*R,D_size{l}*R]);inv_D_precision{l} = (temp*temp');eye(D_size{l}*R);
    temp = randn([D_size{l}*R,1]) + 1i*randn([D_size{l}*R,1]);
    D_mean{l} = temp./abs(temp);
end
x_prior_precision = mat2cell(ones([R*L,1]),[R,R,R],1);
x_precision = diag(ones([R^L,1]));
x_mean = randn([R^L,1]) + 1i*randn([R^L,1]);
n_precision = 1/eps;


% normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
% normalized_error = sum(abs(reshape(double(hosvd(tensor(noisy_channel),2,'rank',[2,2,2])),[],1)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
% [U,~]=cpd((noisy_channel),2);
% normalized_error = sum(abs(sum(khatrirao(U{3},U{2},U{1}),2)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
opts.noARDiter=25;
opts.method='Sparse';
opts.constrFACT=[0 0 0];
opts.constrCore=0;
Temp = zeros([R,R,R]); index = 1:R; Temp(sub2ind(size(Temp),index,index,index))=1;
opts.Core = Temp;
opts.constCore=1;
opts.SNR = SNR;
[~, FACT,~,~] = ARDTUCKER(noisy_channel, [R,R,R], opts);
normalized_error = sum(abs(sum(khatrirao(FACT{3},FACT{2},FACT{1}),2)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');

end


%% Channel estimation using hosvd
function [normalized_error] = hosvd_result(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});

% 
% lena512 = imread('lena512color.tiff');
% lena512 = imresize(lena512,0.125);
% lena512 = double(lena512);
% imshow(uint8(lena512(:,:,2)))


R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[original_channel, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
% Parameter Initialization
D_precision = {}; D_mean = {}; D_shape = {}; D_rate = {};
D_size = {Mx,Mz,N};
for l = 1:L
    D_prior_seperate_precision{l} = ones([R,1]);
    temp = rand([D_size{l},D_size{l}]);D_prior_common_precision{l} = (temp*temp');eye(D_size{l});
    temp = rand([D_size{l}*R,D_size{l}*R]);inv_D_precision{l} = (temp*temp');eye(D_size{l}*R);
    temp = randn([D_size{l}*R,1]) + 1i*randn([D_size{l}*R,1]);
    D_mean{l} = temp./abs(temp);
end
x_prior_precision = mat2cell(ones([R*L,1]),[R,R,R],1);
x_precision = diag(ones([R^L,1]));
x_mean = randn([R^L,1]) + 1i*randn([R^L,1]);
n_precision = 1/eps;


% normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
normalized_error = sum(abs(reshape(double(hosvd(tensor(noisy_channel),2,'rank',[2,2,2])),[],1)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
% [U,~]=cpd((noisy_channel),2);
% normalized_error = sum(abs(sum(khatrirao(U{3},U{2},U{1}),2)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
% opts.noARDiter=25;
% opts.method='Sparse';
% opts.constrFACT=[0 0 0];
% opts.constrCore=0;
% Temp = zeros([R,R,R]); index = 1:R; Temp(sub2ind(size(Temp),index,index,index))=1;
% opts.Core = Temp;
% opts.constCore=1;
% [~, FACT,~,~] = ARDTUCKER(noisy_channel, [R,R,R], opts);
% normalized_error = sum(abs(sum(khatrirao(FACT{3},FACT{2},FACT{1}),2)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');

end

%% Channel estimation using cpd
function [normalized_error] = cpd_result(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});

% 
% lena512 = imread('lena512color.tiff');
% lena512 = imresize(lena512,0.125);
% lena512 = double(lena512);
% imshow(uint8(lena512(:,:,2)))


R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[original_channel, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
% Parameter Initialization
D_precision = {}; D_mean = {}; D_shape = {}; D_rate = {};
D_size = {Mx,Mz,N};
for l = 1:L
    D_prior_seperate_precision{l} = ones([R,1]);
    temp = rand([D_size{l},D_size{l}]);D_prior_common_precision{l} = (temp*temp');eye(D_size{l});
    temp = rand([D_size{l}*R,D_size{l}*R]);inv_D_precision{l} = (temp*temp');eye(D_size{l}*R);
    temp = randn([D_size{l}*R,1]) + 1i*randn([D_size{l}*R,1]);
    D_mean{l} = temp./abs(temp);
end
x_prior_precision = mat2cell(ones([R*L,1]),[R,R,R],1);
x_precision = diag(ones([R^L,1]));
x_mean = randn([R^L,1]) + 1i*randn([R^L,1]);
n_precision = 1/eps;


% normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
% normalized_error = sum(abs(reshape(double(hosvd(tensor(noisy_channel),2,'rank',[2,2,2])),[],1)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
[U,~]=cpd((noisy_channel),2);
normalized_error = sum(abs(sum(khatrirao(U{3},U{2},U{1}),2)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
output = BCPF(noisy_channel, 'maxRank', R, 'dimRed', 1);
normalized_error1 = sum(abs(reshape(double(output.X),[],1)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end

%% Channel estimation using cpd
function [normalized_error] = zhaoqibin_result(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});

% 
% lena512 = imread('lena512color.tiff');
% lena512 = imresize(lena512,0.125);
% lena512 = double(lena512);
% imshow(uint8(lena512(:,:,2)))


R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[original_channel, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
% Parameter Initialization
D_precision = {}; D_mean = {}; D_shape = {}; D_rate = {};
D_size = {Mx,Mz,N};
for l = 1:L
    D_prior_seperate_precision{l} = ones([R,1]);
    temp = rand([D_size{l},D_size{l}]);D_prior_common_precision{l} = (temp*temp');eye(D_size{l});
    temp = rand([D_size{l}*R,D_size{l}*R]);inv_D_precision{l} = (temp*temp');eye(D_size{l}*R);
    temp = randn([D_size{l}*R,1]) + 1i*randn([D_size{l}*R,1]);
    D_mean{l} = temp./abs(temp);
end
x_prior_precision = mat2cell(ones([R*L,1]),[R,R,R],1);
x_precision = diag(ones([R^L,1]));
x_mean = randn([R^L,1]) + 1i*randn([R^L,1]);
n_precision = 1/eps;


% normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
% normalized_error = sum(abs(reshape(double(hosvd(tensor(noisy_channel),2,'rank',[2,2,2])),[],1)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
% [U,~]=cpd((noisy_channel),2);
% normalized_error = sum(abs(sum(khatrirao(U{3},U{2},U{1}),2)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
output = BCPF(noisy_channel, 'maxRank', R, 'init', 'ml', 'dimRed', 0, 'verbose', 0);
normalized_error = sum(abs(reshape(double(output.X),[],1)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end

%% Channel estimation using intra-correlation Tucker SBL
function [normalized_error] = Intra_Correlation_Tucker_SBL(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});

% 
% lena512 = imread('lena512color.tiff');
% lena512 = imresize(lena512,0.125);
% lena512 = double(lena512);
% imshow(uint8(lena512(:,:,2)))


R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[original_channel, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
% Parameter Initialization
D_precision = {}; D_mean = {}; D_shape = {}; D_rate = {};
D_size = {Mx,Mz,N};
for l = 1:L
    D_prior_seperate_precision{l} = ones([R,1]);
    temp = rand([D_size{l},D_size{l}]);D_prior_common_precision{l} = (temp*temp');eye(D_size{l});
    temp = rand([D_size{l}*R,D_size{l}*R]);inv_D_precision{l} = (temp*temp');eye(D_size{l}*R);
    temp = randn([D_size{l}*R,1]) + 1i*randn([D_size{l}*R,1]);
    D_mean{l} = temp./abs(temp);
end
x_prior_precision = mat2cell(ones([R*L,1]),[R,R,R],1);
x_precision = diag(ones([R^L,1]));
x_mean = randn([R^L,1]) + 1i*randn([R^L,1]);
n_precision = 1/eps;


for t = 1:T
    % update D
    for l = 1:L
        % calculate D_mean_square_exclusive
        D_mean_square_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                D_covatiance{i} = cellfun(@sum,cellfun(@diag,mat2cell(inv_D_precision{i}.',ones(R,1)*D_size{i}, ones(R,1)*D_size{i}),'UniformOutput',false));
                D_mean_square_exclusive = kron(D_mean_square_exclusive,(D_covatiance{i}+reshape(D_mean{i},D_size{i},R)'*reshape(D_mean{i},D_size{i},R)));
            end
        end
        % calculate D_precision
        x_mean_tensor = reshape(x_mean,R,R,R); x_mean_index_trnsor = reshape(linspace(1,R^L,R^L),R,R,R);
        x_mean_mode = double(tenmat(x_mean_tensor,l)); x_mean_index_mode = double(tenmat(x_mean_index_trnsor,l));
        x_covariance = pinv(x_precision);
        temp = kron(x_mean_mode,conj(x_mean_mode)) + x_covariance(sub2ind(size(x_covariance),kron(x_mean_index_mode,ones(size(x_mean_index_mode))),kron(ones(size(x_mean_index_mode)),x_mean_index_mode)));
%         D_precision{l} = n_precision*kron(reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l}))+kron(diag(D_prior_seperate_precision{l}),D_prior_common_precision{l});
        % calculate D_mean_exclusive
        D_mean_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                D_mean_exclusive = kron(D_mean_exclusive,reshape(D_mean{i},D_size{i},R));
            end
        end
        % Reduce complexity for inv(D_precision{l})
%         D = kron(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l}));
%         C = kron(diag(D_prior_seperate_precision{l}),D_prior_common_precision{l});
        
%         inv_D_precision{l}=inv(eye(R*D_size{l})+inv(D)*C)*inv(D);

%         inv_D_precision{l}=inv(eye(R*D_size{l})+inv(kron(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l})))*kron(diag(D_prior_seperate_precision{l}),D_prior_common_precision{l})...
%             )*inv(kron(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l})));
%         
%           % Can work
%         inv_D_precision{l}=inv(eye(R*D_size{l})+kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R))*...
%             diag(D_prior_seperate_precision{l}),...
%             D_prior_common_precision{l})...
%             )*inv(kron(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l})));
%         
        


        [Eigenmatrix_P, Eigenvalue_P] = eig(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R))*diag(D_prior_seperate_precision{l}));
%         norm(Eigenmatrix_P*Eigenvalue_P*Eigenmatrix_P'-buzhidao,'fro');
        [Eigenmatrix_Q, Eigenvalue_Q] = eig(D_prior_common_precision{l});
        
%         inv_D_precision{l}=inv(kron(Eigenmatrix_P*Eigenmatrix_P',Eigenmatrix_Q*Eigenmatrix_Q')+...
%             kron(Eigenmatrix_P*Eigenvalue_P*Eigenmatrix_P',Eigenmatrix_Q*Eigenvalue_Q*Eigenmatrix_Q')...
%             )*kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));
%         

%           inv_D_precision{l}=inv(kron(Eigenmatrix_P,Eigenmatrix_Q)*kron(Eigenmatrix_P,Eigenmatrix_Q)'+...
%               kron(Eigenmatrix_P,Eigenmatrix_Q)*kron(Eigenvalue_P,Eigenvalue_Q)*kron(Eigenmatrix_P,Eigenmatrix_Q)'...
%             )*kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));
%        
%          % Can work
%         inv_D_precision{l}=kron(Eigenmatrix_P,Eigenmatrix_Q)*inv(eye(R*D_size{l})+...
%               kron(Eigenvalue_P,Eigenvalue_Q)...
%             )*kron(inv(Eigenmatrix_P),inv(Eigenmatrix_Q))...
%             *kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));

        
        inv_D_precision{l}=kron(Eigenmatrix_P,Eigenmatrix_Q)*diag(...
            1./(1+diag(kron(Eigenvalue_P,Eigenvalue_Q)))...
            )*kron(inv(Eigenmatrix_P),inv(Eigenmatrix_Q))...
            *kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));

        
        
        
        
%         inv_D_precision{l} = inv(eye(R*D_size{l}) + kron((inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R))*diag(D_prior_seperate_precision{l}))...
%        ,D_prior_common_precision{l}));
%         
%         inv_D_precision{l} = kron(Eigenmatrix_P,Eigenmatrix_Q)' * diag(1./(1+diag(kron(Eigenvalue_P,Eigenvalue_Q)))) * kron(Eigenmatrix_P,Eigenmatrix_Q) ...
%             * kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));
%         sum(abs(inv_D_precision{l}-inv(D_precision{l})),'all') / sum(abs(inv(D_precision{l})),'all');
        
        % calculate D_mean
        D_mean{l} = n_precision*inv_D_precision{l}*kron(D_mean_exclusive*x_mean_mode.',eye(D_size{l}))'*reshape(double(tenmat(noisy_channel,l)),[],1);
    end
    
% %     update x % Transpose of D_mean_square 
%     D_mean_square_T = 1;
%     for l = L:-1:1
%         D_mean_square_T = D_mean_square_T.*cellfun(@sum,cellfun(@diag,mat2cell(inv(D_precision{l})+D_mean{l}*D_mean{l}',ones(R,1)*D_size{l}, ones(R,1)*D_size{l}),'UniformOutput',false));
%     end
    % update x
    D_mean_square = 1;
    for l = L:-1:1
        D_covatiance{l} = cellfun(@sum,cellfun(@diag,mat2cell(inv_D_precision{l}.',ones(R,1)*D_size{l}, ones(R,1)*D_size{l}),'UniformOutput',false));
        D_mean_square = kron(D_mean_square,(D_covatiance{l}+reshape(D_mean{l},D_size{l},R)'*reshape(D_mean{l},D_size{l},R)));
    end
    
    % calculate x_precision
    x_prior_precision_temp = 1;
    for l = L:-1:1
        x_prior_precision_temp = kron(x_prior_precision_temp,x_prior_precision{l});
    end
    x_precision = n_precision*D_mean_square+diag(x_prior_precision_temp);
    D_mean_all = 1;
    for l = L:-1:1
        D_mean_all = kron(D_mean_all,reshape(D_mean{l},D_size{l},R));
    end
    % calculate x_mean
    x_mean = n_precision*pinv(x_precision)*D_mean_all'*vectorized_noisy_channel;
    
    % calculate x_prior_precision
    for l = 1:L
        numerator = R^(L-1);
        % calculate x related terms
        x_mean_tensor = reshape(x_mean,R,R,R); x_mean_index_trnsor = reshape(linspace(1,R^L,R^L),R,R,R);
        x_mean_mode = double(tenmat(x_mean_tensor,l)); x_mean_index_mode = double(tenmat(x_mean_index_trnsor,l));
        x_covariance = pinv(x_precision);
        temp = kron(conj(x_mean_mode),x_mean_mode) + x_covariance(sub2ind(size(x_covariance),kron(ones(size(x_mean_index_mode)),x_mean_index_mode),kron(x_mean_index_mode,ones(size(x_mean_index_mode)))));
        % calculate x_prior_precision related terms
        x_prior_precision_temp = 1;
        for i = L:-1:1
            if i~= l
                x_prior_precision_temp = kron(x_prior_precision_temp,x_prior_precision{i});
            end
        end
        denominator = diag(reshape(temp*reshape(diag(x_prior_precision_temp),[],1),R,R));
        x_prior_precision{l} = numerator./denominator;
    end
%     x_prior_precision = 1./diag(pinv(x_precision)+x_mean*x_mean');
    
    % update n_precision
%         D_mean_square0 = 1;
%     for l = L:-1:1
%         D_covatiance{l} = cellfun(@sum,cellfun(@diag,mat2cell(inv_D_precision{l}.',ones(R,1)*D_size{l}, ones(R,1)*D_size{l}),'UniformOutput',false));
%         D_mean_square0 = D_mean_square0.*(D_covatiance{l}+reshape(D_mean{l},D_size{l},R)'*reshape(D_mean{l},D_size{l},R));
%     end
    
    n_precision = prod(cell2mat(D_size)) / ...
        (vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_mean_all'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_mean_all*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(pinv(x_precision).',1,[]))*reshape(D_mean_square,[],1));
    
%     n_precision=prod(cell2mat(D_size)) /((vectorized_noisy_channel-D_mean_all*x_mean)'*(vectorized_noisy_channel-D_mean_all*x_mean)+0);trace(D_mean_all'*D_mean_all*inv(x_precision))
%     n_precision = 1/noise_variance;
    
%     n_precision = prod(cell2mat(D_size))  / ((vectorized_noisy_channel-D_mean_all*x_mean)'*(vectorized_noisy_channel-D_mean_all*x_mean));
    % D_prior_seperate_precision
    for l=1:L
        D_prior_seperate_precision_temp = diag(cellfun(@trace,cellfun(@(x)D_prior_common_precision{l}*x,mat2cell(inv_D_precision{l}+D_mean{l}*D_mean{l}',ones(R,1)*D_size{l}, ones(R,1)*D_size{l}),'UniformOutput',false)));
        D_prior_seperate_precision{l} = D_size{l}./D_prior_seperate_precision_temp;
%         temp(temp<0) = 0;
%         D_prior_seperate_precision{l} = temp;
    end
    
%     % D_prior_common_precision
%     for l=1:L
%         temp_1 = kron(reshape(D_mean{l},D_size{l},R),conj(reshape(D_mean{l},D_size{l},R)));
%         temp_2 = reshape(diag(D_prior_seperate_precision{l}),[],1);
%         D_prior_common_precision{l} = inv(reshape(temp_1*temp_2,D_size{l},D_size{l}).'/R);
%     end
        % D_prior_common_precision
    for l=1:L
        D_prior_common_precision_temp = mat2cell(inv_D_precision{l}+D_mean{l}*D_mean{l}',ones(R,1)*D_size{l}, ones(R,1)*D_size{l});
        D_prior_common_precision_temp_temp = cellfun(@(a,b)a*b,num2cell(D_prior_seperate_precision{l}),D_prior_common_precision_temp(logical(eye(R,R))),'UniformOutput',false);
        D_prior_common_precision{l} = pinv(sum(cat(3,D_prior_common_precision_temp_temp{:}),3).'/R);
        D_prior_common_precision{l} =  D_prior_common_precision{l}/norm(D_prior_common_precision{l},'fro');
%       eye(D_size{l});  D_prior_seperate_precision{l} = D_prior_seperate_precision{l}*norm(D_prior_common_precision{l},'fro');
    end

end
% 1/n_precision > noise_variance;

normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
% sum(abs(reshape(double(hosvd(tensor(noisy_channel),2,'rank',[2,2,2])),[],1)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all')
% [U,~]=cpd((noisy_channel),2);
% sum(abs(sum(khatrirao(U{3},U{2},U{1}),2)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all')

end


%% Channel estimation using intra-correlation CP SBL
function [normalized_error] = Intra_Correlation_CP_SBL(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
% Parameter Initialization
D_precision = {}; D_mean = {}; D_shape = {}; D_rate = {};
D_size = {Mx,Mz,N};
for l = 1:L
    D_prior_seperate_precision{l} = ones([R,1]);
    temp = rand([D_size{l},D_size{l}]);D_prior_common_precision{l} = (temp*temp');eye(D_size{l});
    temp = rand([D_size{l}*R,D_size{l}*R]);D_precision{l} = (temp*temp');eye(D_size{l}*R);
    temp = randn([D_size{l}*R,1]) + 1i*randn([D_size{l}*R,1]);
    D_mean{l} = temp./abs(temp);
end
x_prior_precision = ones([R,1]);
x_precision = diag(ones([R,1]));
x_mean = randn([R,1]) + 1i*randn([R,1]);
n_precision = 1/eps;


for t = 1:T
    % update D
    for l = 1:L
        % calculate D_mean_square_exclusive
        D_mean_square_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                D_covatiance{i} = cellfun(@sum,cellfun(@diag,mat2cell(inv(D_precision{i}).',ones(R,1)*D_size{i}, ones(R,1)*D_size{i}),'UniformOutput',false));
                D_mean_square_exclusive = D_mean_square_exclusive.*(D_covatiance{i}+reshape(D_mean{i},D_size{i},R)'*reshape(D_mean{i},D_size{i},R));
            end
        end
        % calculate D_precision
        temp = diag(kron(x_mean,conj(x_mean))+reshape(inv(x_precision).',[],1));
        D_precision{l} = n_precision*kron(reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l}))+kron(diag(D_prior_seperate_precision{l}),D_prior_common_precision{l});
        % calculate D_mean_exclusive
        D_mean_exclusive = ones(1,R);
        for i = L:-1:1
            if i ~= l
                D_mean_exclusive = khatrirao(D_mean_exclusive,reshape(D_mean{i},D_size{i},R));
            end
        end
        % Reduce complexity for inv(D_precision{l})
%         D = kron(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l}));
%         C = kron(diag(D_prior_seperate_precision{l}),D_prior_common_precision{l});
        
%         inv_D_precision{l}=inv(eye(R*D_size{l})+inv(D)*C)*inv(D);

%         inv_D_precision{l}=inv(eye(R*D_size{l})+inv(kron(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l})))*kron(diag(D_prior_seperate_precision{l}),D_prior_common_precision{l})...
%             )*inv(kron(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l})));
%         
%           % Can work
%         inv_D_precision{l}=inv(eye(R*D_size{l})+kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R))*...
%             diag(D_prior_seperate_precision{l}),...
%             D_prior_common_precision{l})...
%             )*inv(kron(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R),eye(D_size{l})));
%         
        


        [Eigenmatrix_P, Eigenvalue_P] = eig(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R))*diag(D_prior_seperate_precision{l}));
%         norm(Eigenmatrix_P*Eigenvalue_P*Eigenmatrix_P'-buzhidao,'fro');
        [Eigenmatrix_Q, Eigenvalue_Q] = eig(D_prior_common_precision{l});
        
%         inv_D_precision{l}=inv(kron(Eigenmatrix_P*Eigenmatrix_P',Eigenmatrix_Q*Eigenmatrix_Q')+...
%             kron(Eigenmatrix_P*Eigenvalue_P*Eigenmatrix_P',Eigenmatrix_Q*Eigenvalue_Q*Eigenmatrix_Q')...
%             )*kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));
%         

%           inv_D_precision{l}=inv(kron(Eigenmatrix_P,Eigenmatrix_Q)*kron(Eigenmatrix_P,Eigenmatrix_Q)'+...
%               kron(Eigenmatrix_P,Eigenmatrix_Q)*kron(Eigenvalue_P,Eigenvalue_Q)*kron(Eigenmatrix_P,Eigenmatrix_Q)'...
%             )*kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));
%        
%          % Can work
%         inv_D_precision{l}=kron(Eigenmatrix_P,Eigenmatrix_Q)*inv(eye(R*D_size{l})+...
%               kron(Eigenvalue_P,Eigenvalue_Q)...
%             )*kron(inv(Eigenmatrix_P),inv(Eigenmatrix_Q))...
%             *kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));

        
        inv_D_precision{l}=kron(Eigenmatrix_P,Eigenmatrix_Q)*diag(...
            1./(1+diag(kron(Eigenvalue_P,Eigenvalue_Q)))...
            )*kron(inv(Eigenmatrix_P),inv(Eigenmatrix_Q))...
            *kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));

        
        
        
        
%         inv_D_precision{l} = inv(eye(R*D_size{l}) + kron((inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R))*diag(D_prior_seperate_precision{l}))...
%        ,D_prior_common_precision{l}));
%         
%         inv_D_precision{l} = kron(Eigenmatrix_P,Eigenmatrix_Q)' * diag(1./(1+diag(kron(Eigenvalue_P,Eigenvalue_Q)))) * kron(Eigenmatrix_P,Eigenmatrix_Q) ...
%             * kron(inv(n_precision*reshape(temp*reshape(D_mean_square_exclusive,[],1),R,R)),eye(D_size{l}));
%         sum(abs(inv_D_precision{l}-inv(D_precision{l})),'all') / sum(abs(inv(D_precision{l})),'all');
        
        % calculate D_mean
        D_mean{l} = n_precision*inv_D_precision{l}*kron(D_mean_exclusive*diag(x_mean),eye(D_size{l}))'*reshape(double(tenmat(noisy_channel,l)),[],1);
    end
    
% %     update x % Transpose of D_mean_square 
%     D_mean_square_T = 1;
%     for l = L:-1:1
%         D_mean_square_T = D_mean_square_T.*cellfun(@sum,cellfun(@diag,mat2cell(inv(D_precision{l})+D_mean{l}*D_mean{l}',ones(R,1)*D_size{l}, ones(R,1)*D_size{l}),'UniformOutput',false));
%     end
    % update x
    D_mean_square = 1;
    for l = L:-1:1
        D_covatiance{l} = cellfun(@sum,cellfun(@diag,mat2cell(inv_D_precision{l}.',ones(R,1)*D_size{l}, ones(R,1)*D_size{l}),'UniformOutput',false));
        D_mean_square = D_mean_square.*(D_covatiance{l}+reshape(D_mean{l},D_size{l},R)'*reshape(D_mean{l},D_size{l},R));
    end
    
    % calculate x_precision
    x_precision = n_precision*D_mean_square+diag(x_prior_precision);
    D_mean_all = ones(1,R);
    for l = L:-1:1
        D_mean_all = khatrirao(D_mean_all,reshape(D_mean{l},D_size{l},R));
    end
    % calculate x_mean
    x_mean = n_precision*inv(x_precision)*D_mean_all'*vectorized_noisy_channel;
    
    % calculate x_prior_precision
    x_prior_precision = 1./diag(inv(x_precision)+x_mean*x_mean');
    
    % update n_precision
%         D_mean_square = 1;
%     for l = L:-1:1
%         D_covatiance{l} = cellfun(@sum,cellfun(@diag,mat2cell(inv_D_precision{l}.',ones(R,1)*D_size{l}, ones(R,1)*D_size{l}),'UniformOutput',false));
%         D_mean_square = D_mean_square.*(D_covatiance{l}+reshape(D_mean{l},D_size{l},R)'*reshape(D_mean{l},D_size{l},R));
%     end
    
    n_precision = prod(cell2mat(D_size)) / ...
        (vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_mean_all'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_mean_all*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(D_mean_square,[],1));
    
%     n_precision=prod(cell2mat(D_size)) /((vectorized_noisy_channel-D_mean_all*x_mean)'*(vectorized_noisy_channel-D_mean_all*x_mean)+0);trace(D_mean_all'*D_mean_all*inv(x_precision))
%     n_precision = 1/noise_variance;
    
%     n_precision = prod(cell2mat(D_size))  / ((vectorized_noisy_channel-D_mean_all*x_mean)'*(vectorized_noisy_channel-D_mean_all*x_mean));
    % D_prior_seperate_precision
    for l=1:L
        D_prior_seperate_precision_temp = diag(cellfun(@trace,cellfun(@(x)D_prior_common_precision{l}*x,mat2cell(inv(D_precision{l})+D_mean{l}*D_mean{l}',ones(R,1)*D_size{l}, ones(R,1)*D_size{l}),'UniformOutput',false)));
        D_prior_seperate_precision{l} = D_size{l}./D_prior_seperate_precision_temp;
%         temp(temp<0) = 0;
%         D_prior_seperate_precision{l} = temp;
    end
    
%     % D_prior_common_precision
%     for l=1:L
%         temp_1 = kron(reshape(D_mean{l},D_size{l},R),conj(reshape(D_mean{l},D_size{l},R)));
%         temp_2 = reshape(diag(D_prior_seperate_precision{l}),[],1);
%         D_prior_common_precision{l} = inv(reshape(temp_1*temp_2,D_size{l},D_size{l}).'/R);
%     end
        % D_prior_common_precision
    for l=1:L
        D_prior_common_precision_temp = mat2cell(inv_D_precision{l}+D_mean{l}*D_mean{l}',ones(R,1)*D_size{l}, ones(R,1)*D_size{l});
        D_prior_common_precision_temp_temp = cellfun(@(a,b)a*b,num2cell(D_prior_seperate_precision{l}),D_prior_common_precision_temp(logical(eye(R,R))),'UniformOutput',false);
        D_prior_common_precision{l} = pinv(sum(cat(3,D_prior_common_precision_temp_temp{:}),3).'/R);
        D_prior_common_precision{l} =  D_prior_common_precision{l}/norm(D_prior_common_precision{l},'fro');
%      temp = ones(D_size{L})*0.9; temp(logical((eye(D_size{L})))) = ones(D_size{L},1);
%      eye(D_size{l});  D_prior_common_precision{L} = temp;           
        
%         D_prior_seperate_precision{l} = D_prior_seperate_precision{l}*norm(D_prior_common_precision{l},'fro');
    end

end
% 1/n_precision > noise_variance;

normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


%% Channel estimation using Traditional full SBL DL
function [normalized_error] = Traditional_FULL_SBL_DL(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, ~, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
% Parameter Initialization
D_size = {Mx,Mz,N};
D_shape_prior = ones([R,1])*eps;
D_rate_prior = ones([R,1])*eps;
D_precision = diag(ones([R,1]));
temp = randn([Mx*Mz*N,R]) + 1i*randn([Mx*Mz*N,R]);
D_mean = temp./abs(temp);
x_shape_prior = ones([R,1])*eps;
x_rate_prior = ones([R,1])*eps;
x_precision = diag(ones([R,1]));
x_mean = randn([R,1]) + 1i*randn([R,1]);
n_shape_prior = eps;
n_rate_prior = eps;
% Initialization
D_shape = ones([R,1])*eps;
D_rate = ones([R,1])*eps;
x_shape = ones([R,1])*eps;
x_rate = ones([R,1])*eps;
n_shape = 1;
n_rate = sqrt(eps);

for t = 1:T
    
    % update D
    % calculate D_precision
    D_precision = n_shape/n_rate*(inv(x_precision)+x_mean*x_mean')+diag(D_shape./D_rate);
    % calculate D_mean
    D_mean = n_shape/n_rate*vectorized_noisy_channel*x_mean'*inv(D_precision);
    % calculate D_shape and D_rate
    D_shape = Mx*Mz*N+D_shape_prior;
    D_rate = diag(prod(cell2mat(D_size))*inv(D_precision)+D_mean'*D_mean)+D_rate_prior;
    
    % updata x
    D_mean_square = prod(cell2mat(D_size))*inv(D_precision)+D_mean'*D_mean;
    % calculate x_precision
    x_precision = n_shape/n_rate*D_mean_square+diag(x_shape./x_rate);
    % calculate x_mean
    x_mean = n_shape/n_rate*inv(x_precision)*D_mean'*vectorized_noisy_channel;
    % calculate x_shape annd x_rate
    x_shape = 1+x_shape_prior;
    x_mean_square = inv(x_precision)+x_mean*x_mean';
    x_rate = diag(x_mean_square)+x_rate_prior;

    % update n
    % calculate n_shape annd n_rate
    n_shape = prod(cell2mat(D_size))+n_shape_prior;
    n_rate = vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_mean'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_mean*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(D_mean_square,[],1) + n_rate_prior; 
    
end
normalized_error = sum(abs(D_mean*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


%% Channel estimation using CP based full SBL DL
function [normalized_error] = CP_FULL_SBL_DL(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
fake_noise_variance = mean(abs(vectorized_noisy_channel).^2)/100;
% Parameter Initialization
D_size = {Mx,Mz,N};
for l = 1:L
    D_shape_prior{l} = ones([R,1])*eps;
    D_rate_prior{l} = ones([R,1])*eps;
    D_precision{l} = diag(ones([R,1]));
    temp = randn([D_size{l},R]) + 1i*randn([D_size{l},R]);
    D_mean{l} = temp./abs(temp);
end
x_shape_prior = ones([R,1])*eps;
x_rate_prior = ones([R,1])*eps;
x_precision = diag(ones([R,1]));
x_mean = randn([R,1]) + 1i*randn([R,1]);
n_shape_prior = eps;
n_rate_prior = eps;
% Initialization
D_shape = D_shape_prior;
D_rate = D_rate_prior;
x_shape = x_shape_prior;
x_rate = x_rate_prior;
n_shape = 1;
n_rate = eps;

for t = 1:T
    
    
%         D_p = D_shape{1}./D_rate{1};
%     pp = D_p > 50000;
%     
%     x_shape(pp) = [];
%     x_rate(pp) = [];
%     x_mean(pp) = [];
%     x_shape_prior(pp) = [];
%     x_rate_prior(pp) = [];
%     x_precision(:,pp) = [];
%     x_precision(pp,:) = [];
%     
%     for l=1:L
%         d1 = D_mean{l};
%         d1(:,pp) = [];
%         D_mean{l} = d1;
%         
%         d1 = D_shape{l};
%         d1(pp) = [];
%         D_shape{l} = d1;
%         
%         d1 = D_rate{l};
%         d1(pp) = [];
%         D_rate{l} = d1;
%         
%         d1 = D_shape_prior{l};
%         d1(pp) = [];
%         D_shape_prior{l} = d1;
%         
%         d1 = D_rate_prior{l};
%         d1(pp) = [];
%         D_rate_prior{l} = d1;
%         
%         d1 = D_precision{l};
%         d1(:,pp) = [];
%         d1(pp,:) = [];
%         D_precision{l} = d1;
%     end
%     
%     R = R -sum(pp);
    
    
    
    % update D
    for l = 1:L
        % calculate D_mean_square_exclusive
        D_mean_square_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                D_mean_square_exclusive = D_mean_square_exclusive.*(D_size{i}*inv(D_precision{i})+D_mean{i}'*D_mean{i});
            end
        end
        % calculate D_precision
        temp = diag(kron(conj(x_mean),x_mean)+reshape(inv(x_precision),[],1));
        D_precision{l} = n_shape/n_rate*reshape(temp*reshape(conj(D_mean_square_exclusive),[],1),R,R)+diag(D_shape{l}./D_rate{l});
        % calculate D_mean_exclusive
        D_mean_exclusive = ones(1,R);
        for i = L:-1:1
            if i ~= l
                D_mean_exclusive = khatrirao(D_mean_exclusive,conj(D_mean{i}));
            end
        end
        % calculate D_mean
        D_mean{l} = n_shape/n_rate*double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean')*inv(D_precision{l});

        % calculate D_shape and D_rate
        D_shape{l} = D_size{l}+D_shape_prior{l};
        D_rate{l} = diag(D_size{l}*inv(D_precision{l})+D_mean{l}'*D_mean{l})+D_rate_prior{l};
        
    end
    
    % updata x
    D_mean_square = 1;
    for l = L:-1:1
        D_mean_square = D_mean_square.*(D_size{l}*inv(D_precision{l})+D_mean{l}'*D_mean{l});
    end
    % calculate x_precision
    x_precision = n_shape/n_rate*(D_mean_square+diag(x_shape./x_rate));
    D_meanranspose = ones(1,R);
    for l = L:-1:1
        D_meanranspose = khatrirao(D_meanranspose,D_mean{l});
    end
    % calculate x_mean
    x_mean = n_shape/n_rate*inv(x_precision)*D_meanranspose'*vectorized_noisy_channel;
    % calculate x_shape annd x_rate
    x_shape = 1+x_shape_prior;
    x_mean_square = inv(x_precision)+x_mean*x_mean';
    x_rate = n_shape/n_rate*diag(x_mean_square)+x_rate_prior;
    
    % update n
    % calculate n_shape annd n_rate
    n_shape = prod(cell2mat(D_size))+R+n_shape_prior;
    n_rate = vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_meanranspose'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_meanranspose*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(D_mean_square,[],1) ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(diag(x_shape./x_rate),[],1) + n_rate_prior; 
    
%     n_rate = vectorized_noisy_channel'*inv(eye(prod(cell2mat(D_size)))+D_meanranspose*diag(x_rate./x_shape)*D_meanranspose')*vectorized_noisy_channel;

%     n_rate = (vectorized_noisy_channel-D_meanranspose*x_mean)'*(vectorized_noisy_channel-D_meanranspose*x_mean) + n_rate_prior;

    


% n_shape = 1;
% n_rate = noise_variance;
    
    
    
end
normalized_error = sum(abs(D_meanranspose*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


%% Channel estimation using CP based EM SBL DL
function [normalized_error] = CP_EM_SBL_DL(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
fake_noise_variance = mean(abs(vectorized_noisy_channel).^2)/100;
% Parameter Initialization
D_size = {Mx,Mz,N};
for l = 1:L
    D_shape_prior{l} = ones([R,1])*eps;
    D_rate_prior{l} = ones([R,1])*eps;
    D_precision{l} = diag(ones([R,1]))/eps;
    temp = randn([D_size{l},R]) + 1i*randn([D_size{l},R]);
    D_mean{l} = temp./abs(temp);
end
x_shape_prior = ones([R,1])*2;
x_rate_prior = ones([R,1])*eps;
x_precision = diag(ones([R,1]))/eps;
x_mean = randn([R,1]) + 1i*randn([R,1]);
n_shape_prior = 1;
n_rate_prior = eps;
% Initialization
D_shape = D_shape_prior;
D_rate = D_rate_prior;
x_shape = x_rate_prior;
x_rate = x_rate_prior;
n_shape = 1;
n_rate = n_rate_prior;

for t = 1:T
    % update D
    for l = 1:L
        % calculate D_mean_square_exclusive
        D_mean_square_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                D_mean_square_exclusive = D_mean_square_exclusive.*(inv(D_precision{i})+D_mean{i}'*D_mean{i});
            end
        end
        % calculate D_precision
        temp = diag(kron(conj(x_mean),x_mean)+reshape(inv(x_precision),[],1));
        D_precision{l} = n_shape/n_rate*reshape(temp*reshape(conj(D_mean_square_exclusive),[],1),R,R)+diag(D_shape{l}./D_rate{l});
        % calculate D_mean_exclusive
        D_mean_exclusive = ones(1,R);
        for i = L:-1:1
            if i ~= l
                D_mean_exclusive = khatrirao(D_mean_exclusive,conj(D_mean{i}));
            end
        end
        % calculate D_mean
        D_mean{l} = n_shape/n_rate*double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean')*inv(D_precision{l});
        
        % calculate D_shape and D_rate
        D_shape{l} = D_size{l}+D_shape_prior{l}-1;
        D_rate{l} = diag(inv(D_precision{l})+D_mean{l}'*D_mean{l})+D_rate_prior{l};
    end
    
    % updata x
    D_mean_square = 1;
    for l = L:-1:1
        D_mean_square = D_mean_square.*(inv(D_precision{l})+D_mean{l}'*D_mean{l});
    end
    % calculate x_precision
    x_precision = n_shape/n_rate*(D_mean_square+diag(x_shape./x_rate));
    D_meanranspose = ones(1,R);
    for l = L:-1:1
        D_meanranspose = khatrirao(D_meanranspose,D_mean{l});
    end
    % calculate x_mean
    x_mean = n_shape/n_rate*inv(x_precision)*D_meanranspose'*vectorized_noisy_channel;
    % calculate x_shape annd x_rate
    x_shape = x_shape_prior;
    x_mean_square = inv(x_precision)+x_mean*x_mean';
    x_rate = n_shape/n_rate*diag(x_mean_square) + x_rate_prior;
    
    % update n
    % calculate n_shape annd n_rate
    n_shape = prod(cell2mat(D_size))+n_shape_prior-1;
    n_rate = vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_meanranspose'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_meanranspose*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(D_mean_square,[],1) ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(diag(x_shape./x_rate),[],1) + n_rate_prior;
% n_rate = vectorized_noisy_channel'*inv(eye(prod(cell2mat(D_size)))+D_meanranspose*diag(x_rate./x_shape)*D_meanranspose')*vectorized_noisy_channel;

n_rate = (vectorized_noisy_channel-D_meanranspose*x_mean)'*(vectorized_noisy_channel-D_meanranspose*x_mean) + n_rate_prior;

end
normalized_error = sum(abs(D_meanranspose*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


%% Channel estimation using CP based full original SBL DL
function [normalized_error] = CP_FULL_original_SBL_DL(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
% Parameter Initialization
D_size = {Mx,Mz,N};
for l = 1:L
    D_shape_prior{l} = ones([R,1])*eps;
    D_rate_prior{l} = ones([R,1])*eps;
    D_precision{l} = diag(ones([R,1]));
    temp = randn([D_size{l},R]) + 1i*randn([D_size{l},R]);
    D_mean{l} = temp./abs(temp);
end
x_shape_prior = ones([R,1])*eps;
x_rate_prior = ones([R,1])*eps;
x_precision = diag(ones([R,1]));
x_mean = randn([R,1]) + 1i*randn([R,1]);
n_shape_prior = eps;
n_rate_prior = eps;
% Initialization
for l = 1:L
    D_shape{l} = ones([R,1])*eps;
    D_rate{l} = ones([R,1])*eps;
end
x_shape = ones([R,1])*eps;
x_rate = ones([R,1])*eps;
n_shape = 1;
n_rate = eps;

for t = 1:T
    
    % update D
    for l = 1:L
        % calculate D_mean_square_exclusive
        D_mean_square_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                D_mean_square_exclusive = D_mean_square_exclusive.*(D_size{i}*inv(D_precision{i})+D_mean{i}'*D_mean{i});
            end
        end
        % calculate D_precision
        temp = diag(kron(conj(x_mean),x_mean)+reshape(inv(x_precision),[],1));
        D_precision{l} = n_shape/n_rate*reshape(temp*reshape(conj(D_mean_square_exclusive),[],1),R,R)+diag(D_shape{l}./D_rate{l});
        % calculate D_mean_exclusive
        D_mean_exclusive = ones(1,R);
        for i = L:-1:1
            if i ~= l
                D_mean_exclusive = khatrirao(D_mean_exclusive,conj(D_mean{i}));
            end
        end
        % calculate D_mean
        D_mean{l} = n_shape/n_rate*double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean')*inv(D_precision{l});
        % calculate D_shape and D_rate
        D_shape{l} = D_size{l}+D_shape_prior{l};
        D_rate{l} = diag(D_size{l}*inv(D_precision{l})+D_mean{l}'*D_mean{l})+D_rate_prior{l};
    end
    
    % updata x
    D_mean_square = 1;
    for l = L:-1:1
        D_mean_square = D_mean_square.*(D_size{l}*inv(D_precision{l})+D_mean{l}'*D_mean{l});
    end
    % calculate x_precision
    x_precision = n_shape/n_rate*D_mean_square+diag(x_shape./x_rate);
    D_mean_all = ones(1,R);
    for l = L:-1:1
        D_mean_all = khatrirao(D_mean_all,D_mean{l});
    end
    % calculate x_mean
    x_mean = n_shape/n_rate*inv(x_precision)*D_mean_all'*vectorized_noisy_channel;
    % calculate x_shape annd x_rate
    x_shape = 1+x_shape_prior;
    x_mean_square = inv(x_precision)+x_mean*x_mean';
    x_rate = diag(x_mean_square)+x_rate_prior;
    
    % update n
    n_shape = prod(cell2mat(D_size))+n_shape_prior;
    n_rate = vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_mean_all'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_mean_all*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(D_mean_square,[],1) + n_rate_prior;
    
end
normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');

% sum(abs(reshape(double(hosvd(tensor(noisy_channel),2,'rank',[2,2,2])),[],1)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all')
% [U,~]=cpd((noisy_channel),2);
% sum(abs(sum(khatrirao(U{3},U{2},U{1}),2)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all')

end


%% Channel estimation using CP based full original SBL DL
function [normalized_error] = CP_FULL_original_SBL_DL_Mixed(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, SNR, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
fake_noise_variance = mean(abs(vectorized_noisy_channel).^2)/100;
% Parameter Initialization
D_size = {Mx,Mz,N};
for l = 1:L
    D_shape_prior{l} = ones([R,1])*eps;
    D_rate_prior{l} = ones([R,1])*eps;
    D_precision{l} = diag(ones([R,1]));
    if l == 3
        temp = randn([D_size{l},R]) + 1i*randn([D_size{l},R]);
        D_mean{l} = temp./abs(temp);
    else
        temp = randn([D_size{l},R]) + 1i*randn([D_size{l},R]);
        [U,~,V] = svd(double(tenmat(noisy_channel,l)), 'econ');%svd(temp, 'econ');%%svd(temp, 'econ');
        D_mean{l} = U(:,1:R);%U*V';%V(1:R,:)';%;%
    end
end
x_shape_prior = ones([R,1])*eps;
x_rate_prior = ones([R,1])*eps;
x_precision = diag(ones([R,1]));
x_mean = randn([R,1]) + 1i*randn([R,1]);
n_shape_prior = eps;
n_rate_prior = eps;
% Initialization
for l = 1:L
    D_shape{l} = ones([R,1])*eps;
    D_rate{l} = ones([R,1])*eps;
end
x_shape = ones([R,1])*eps;
x_rate = ones([R,1])*eps;
n_shape = 1;
n_rate = eps;

for t = 1:T
    
    % update D
    for l = 1:L
        % calculate D_mean_square_exclusive
        D_mean_square_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                if i == 3
                    D_mean_square_exclusive = D_mean_square_exclusive.*(D_size{i}*inv(D_precision{i})+D_mean{i}'*D_mean{i});
                else
                    D_mean_square_exclusive = D_mean_square_exclusive.*(D_mean{i}'*D_mean{i});
                end
            end
        end
        % calculate D_precision
        temp = diag(kron(conj(x_mean),x_mean)+reshape(inv(x_precision),[],1));
        D_precision{l} = n_shape/n_rate*reshape(temp*reshape(conj(D_mean_square_exclusive),[],1),R,R)+diag(D_shape{l}./D_rate{l});
        % calculate D_mean_exclusive
        D_mean_exclusive = ones(1,R);
        for i = L:-1:1
            if i ~= l
                D_mean_exclusive = khatrirao(D_mean_exclusive,conj(D_mean{i}));
            end
        end
        if l == 3
            % calculate D_mean
            D_mean{l} = n_shape/n_rate*double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean')*inv(D_precision{l});
            % calculate D_shape and D_rate
            D_shape{l} = D_size{l}+D_shape_prior{l};
            D_rate{l} = diag(D_size{l}*inv(D_precision{l})+D_mean{l}'*D_mean{l})+D_rate_prior{l};
        else
            % calculate D_mean
            Mises_Fisher{l} = n_shape/n_rate*double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean');
            [U_matrix{l},S,V_matrix{l}] = svd(Mises_Fisher{l},'econ');
            D_mean{l} = U_matrix{l}*V_matrix{l}';
            % calculate D_shape and D_rate
            D_shape{l} = D_size{l}+D_shape_prior{l};
            D_rate{l} = diag(D_mean{l}'*D_mean{l})+D_rate_prior{l};
        end
%         D_shape_T = 0; D_rate_T = 0;
%         if l == 3
%             % calculate D_shape and D_rate
%             D_shape_T = D_shape_T + D_size{l};
%             D_rate_T = diag(D_size{l}*inv(D_precision{l})+D_mean{l}'*D_mean{l});
%         else
%             
%         end
        
        
    end
    
    
    
    % updata x
    D_mean_square = 1;
    for l = L:-1:1
        if l == 3
            D_mean_square = D_mean_square.*(D_size{l}*inv(D_precision{l})+D_mean{l}'*D_mean{l});
        else
            D_mean_square = D_mean_square.*(D_mean{l}'*D_mean{l});
        end
    end
    % calculate x_precision
    x_precision = n_shape/n_rate*D_mean_square+diag(x_shape./x_rate);
    D_mean_all = ones(1,R);
    for l = L:-1:1
        D_mean_all = khatrirao(D_mean_all,D_mean{l});
    end
    % calculate x_mean
    x_mean = n_shape/n_rate*inv(x_precision)*D_mean_all'*vectorized_noisy_channel;
    % calculate x_shape annd x_rate
    x_shape = 1+x_shape_prior;
    x_mean_square = inv(x_precision)+x_mean*x_mean';
    x_rate = diag(x_mean_square)+x_rate_prior;
    
    
%     % Fast algorithm for x_shape and x_rate
%     [~,index] = sort(x_rate,'descend');
%     for index_r = 1:R
%         r = index(index_r);
%         exclusive_x_shape_x_rate = diag(x_shape./x_rate); exclusive_x_shape_x_rate(r,r) = 0;
%         x_precision_exclusive = n_shape/n_rate*D_mean_square+exclusive_x_shape_x_rate;
%         exclusive_vector = zeros(R,1); exclusive_vector(r) = 1;
%         sl(r) = exclusive_vector'*inv(x_precision_exclusive)*exclusive_vector;
%         wl2(r) = (n_shape/n_rate)^2*exclusive_vector'*inv(x_precision_exclusive)*...
%             D_mean_all'*vectorized_noisy_channel*vectorized_noisy_channel'*D_mean_all*...
%             inv(x_precision_exclusive)*exclusive_vector;
%         if wl2(r)-sl(r)>0
%             x_rate(r) = wl2(r)-sl(r);
%         else
%             x_rate(r) = eps;
%         end
%     end
    
    % update n
    n_shape = prod(cell2mat(D_size))+n_shape_prior;
    n_rate = vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_mean_all'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_mean_all*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(D_mean_square,[],1) + n_rate_prior;
    
    n_shape = 1;
    n_rate = noise_variance;
    
end
normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


%% Channel estimation using CP based full original SBL DL
function [normalized_error] = CP_FULL_original_SBL_DL_Orthogonal(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
fake_noise_variance = mean(abs(vectorized_noisy_channel).^2)/100;
% Parameter Initialization
D_size = {Mx,Mz,N};
for l = 1:L
    D_shape_prior{l} = ones([R,1])*eps;
    D_rate_prior{l} = ones([R,1])*eps;
    D_precision{l} = diag(ones([R,1]));
    temp = randn([D_size{l},R]) + 1i*randn([D_size{l},R]);
    D_mean{l} = temp./abs(temp);
end
x_shape_prior = ones([R,1])*eps;
x_rate_prior = ones([R,1])*eps;
x_precision = diag(ones([R,1]));
x_mean = randn([R,1]) + 1i*randn([R,1]);
n_shape_prior = eps;
n_rate_prior = eps;
% Initialization
for l = 1:L
    D_shape{l} = ones([R,1])*eps;
    D_rate{l} = ones([R,1])*eps;
end
x_shape = ones([R,1])*eps;
x_rate = ones([R,1])*eps;
n_shape = 1;
n_rate = noise_variance;

for t = 1:T
    
    % update D
    for l = 1:L
        % calculate D_mean_square_exclusive
        D_mean_square_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                D_mean_square_exclusive = D_mean_square_exclusive.*(D_mean{i}'*D_mean{i});
            end
        end
        % calculate D_precision
        temp = diag(kron(conj(x_mean),x_mean)+reshape(inv(x_precision),[],1));
        D_precision{l} = n_shape/n_rate*reshape(temp*reshape(conj(D_mean_square_exclusive),[],1),R,R);1+diag(D_shape{l}./D_rate{l});
        % calculate D_mean_exclusive
        D_mean_exclusive = ones(1,R);
        for i = L:-1:1
            if i ~= l
                D_mean_exclusive = khatrirao(D_mean_exclusive,conj(D_mean{i}));
            end
        end
        % calculate D_mean
        Mises_Fisher{l} = n_shape/n_rate*double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean');
        [U_matrix{l},S,V_matrix{l}] = svd(Mises_Fisher{l},'econ');
        D_mean{l} = U_matrix{l}*V_matrix{l}';
%         D_mean{l} = n_shape/n_rate*double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean')*inv(D_precision{l});
        % calculate D_shape and D_rate
        D_shape{l} = D_size{l}+D_shape_prior{l};
        D_rate{l} = diag(D_mean{l}'*D_mean{l})+D_rate_prior{l};
    end
    
    
    % updata x
    D_mean_square = 1;
    for l = L:-1:1
        D_mean_square = D_mean_square.*(D_mean{l}'*D_mean{l});
    end
    % calculate x_precision
    x_precision = n_shape/n_rate*D_mean_square+diag(x_shape./x_rate);
    D_mean_all = ones(1,R);
    for l = L:-1:1
        D_mean_all = khatrirao(D_mean_all,D_mean{l});
    end
    % calculate x_mean
    x_mean = n_shape/n_rate*inv(x_precision)*D_mean_all'*vectorized_noisy_channel;
    % calculate x_shape annd x_rate
    x_shape = 1+x_shape_prior;
    x_mean_square = inv(x_precision)+x_mean*x_mean';
    x_rate = diag(x_mean_square)+x_rate_prior;

%     %Fast algorithm for x_shape and x_rate
%     [~,index] = sort(x_rate,'descend');
%     for index_r = 1:R
%         r = index(index_r);
%         exclusive_x_shape_x_rate = diag(x_shape./x_rate); exclusive_x_shape_x_rate(r,r) = 0;
%         x_precision_exclusive = n_shape/n_rate*D_mean_square+exclusive_x_shape_x_rate;
%         exclusive_vector = zeros(R,1); exclusive_vector(r) = 1;
%         sl(r) = exclusive_vector'*inv(x_precision_exclusive)*exclusive_vector;
%         wl2(r) = (n_shape/n_rate)^2*exclusive_vector'*inv(x_precision_exclusive)*...
%             D_mean_all'*vectorized_noisy_channel*vectorized_noisy_channel'*D_mean_all*...
%             inv(x_precision_exclusive)*exclusive_vector;
%         if wl2(r)-sl(r)>0
%             x_rate(r) = wl2(r)-sl(r);
%         else
%             x_rate(r) = eps;
%         end
%     end

    
    % update n
    n_shape = prod(cell2mat(D_size))+n_shape_prior;
    n_rate = vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_mean_all'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_mean_all*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(D_mean_square,[],1) + n_rate_prior;


n_shape = 1;
n_rate = noise_variance;


    
%         % prune zero variables
%     prune_position = (x_mean==0);
%     % x related
%     x_mean(prune_position) = [];
%     x_precision(prune_position,:) = []; x_precision(:,prune_position) = [];
%     x_shape(prune_position) = [];
%     x_rate(prune_position) = [];
%     % D related
%     for l = 1:L
%         D_mean_temp = D_mean{l};
%         D_mean_temp(:,prune_position) = [];
%         D_mean{l} = D_mean_temp;
%         
%         D_precision_temp = D_precision{l};
%         D_precision_temp(prune_position,:)=[];D_precision_temp(:,prune_position)=[];
%         D_precision{l}=D_precision_temp;
%         
%         D_shape_temp = D_shape{l};
%         D_shape_temp(prune_position)=[];
%         D_shape{l}=D_shape_temp;
%         
%         D_rate_temp = D_shape{l};
%         D_rate_temp(prune_position)=[];
%         D_rate{l}=D_rate_temp;
%     end
%     R = R - sum(prune_position);
% for l = 1:L
%     D_shape_prior{l} = ones([R,1])*eps;
%     D_rate_prior{l} = ones([R,1])*eps;
% end
%     x_shape_prior = ones([R,1])*eps;
%     x_rate_prior = ones([R,1])*eps;
    
 
% sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all')
end
normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


%% Channel estimation using CP based full original SBL DL
function [normalized_error] = CP_EM_original_SBL_DL_pppp(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
fake_noise_variance = mean(abs(vectorized_noisy_channel).^2)/100;
% Parameter Initialization
D_size = {Mx,Mz,N};
for l = 1:L
    D_shape_prior{l} = ones([R,1])*eps;
    D_rate_prior{l} = ones([R,1])*eps;
    D_precision{l} = diag(ones([R,1]));
    temp = randn([D_size{l},R]) + 1i*randn([D_size{l},R]);
    D_mean{l} = temp./abs(temp);
end
x_shape_prior = ones([R,1])*eps;
x_rate_prior = ones([R,1])*eps;
x_precision = diag(ones([R,1]));
x_mean = randn([R,1]) + 1i*randn([R,1]);
n_shape_prior = eps;
n_rate_prior = eps;
% Initialization
for l = 1:L
    D_shape{l} = ones([R,1])*eps;
    D_rate{l} = ones([R,1])*eps;
end
x_shape = ones([R,1])*eps;
x_rate = ones([R,1])*eps;
n_shape = 1;
n_rate = eps;

for t = 1:T
    
    % update D
    for l = 1:L
        % calculate D_mean_square_exclusive
        D_mean_square_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                D_mean_square_exclusive = D_mean_square_exclusive.*(D_size{i}*inv(D_precision{i})+D_mean{i}'*D_mean{i});
            end
        end
        % calculate D_precision
        temp = diag(kron(conj(x_mean),x_mean)+reshape(inv(x_precision),[],1));
        D_precision{l} = n_shape/n_rate*reshape(temp*reshape(conj(D_mean_square_exclusive),[],1),R,R);
        % calculate D_mean_exclusive
        D_mean_exclusive = ones(1,R);
        for i = L:-1:1
            if i ~= l
                D_mean_exclusive = khatrirao(D_mean_exclusive,conj(D_mean{i}));
            end
        end
        % calculate D_mean
        D_mean{l} = n_shape/n_rate*double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean')*inv(D_precision{l});
    end
    
    
    % updata x
    D_mean_square = 1;
    for l = L:-1:1
        D_mean_square = D_mean_square.*(D_size{l}*inv(D_precision{l})+D_mean{l}'*D_mean{l});
    end
    % calculate x_precision
    x_precision = n_shape/n_rate*D_mean_square+diag(x_shape./x_rate);
    D_mean_all = ones(1,R);
    for l = L:-1:1
        D_mean_all = khatrirao(D_mean_all,D_mean{l});
    end
    % calculate x_mean
    x_mean = n_shape/n_rate*inv(x_precision)*D_mean_all'*vectorized_noisy_channel;
    % calculate x_shape annd x_rate
    x_shape = 1+x_shape_prior;
    x_mean_square = inv(x_precision)+x_mean*x_mean';
    x_rate = diag(x_mean_square)+x_rate_prior;

%     posi = D_rate{1}<1e-6; 
%     x_rate(posi) = eps;
    
%     x_shape = (1-x_shape./x_rate.*diag(inv(x_precision)));
%     x_rate = diag(x_mean*x_mean')+x_rate_prior;
    
    % update n
    n_shape = prod(cell2mat(D_size))+n_shape_prior;
    n_rate = vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_mean_all'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_mean_all*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(D_mean_square,[],1) + n_rate_prior;

% n_rate=(vectorized_noisy_channel-D_mean_all*x_mean)'*(vectorized_noisy_channel-D_mean_all*x_mean)+trace(D_mean_all'*D_mean_all*inv(x_precision));
% n_shape = 1;
% n_rate = noise_variance;

%     r = (1-x_shape./x_rate.*diag(inv(x_precision)));
%     n_rate = (vectorized_noisy_channel-D_mean_all*x_mean)'*(vectorized_noisy_channel-D_mean_all*x_mean) + n_rate_prior;
end
normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


%% Channel estimation using CP based EM original SBL DL
function [normalized_error] = CP_EM_original_SBL_DL(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
fake_noise_variance = mean(abs(vectorized_noisy_channel).^2)/100;
% Parameter Initialization
D_size = {Mx,Mz,N};
D_mean = {};
for l = 1:L
    temp = randn([D_size{l},R]) + 1i*randn([D_size{l},R]);
%   D_mean{l} = temp./abs(temp);
    [U,~,V] = svd(temp, 'econ');
    D_mean{l} = U*V';
end
x_shape_prior = ones([R,1])*1;
x_rate_prior = ones([R,1])*eps;
x_mean = randn([R,1]) + 1i*randn([R,1]);
x_precision = diag(1);
n_shape_prior = eps;
n_rate_prior = eps;
% Initialization
x_shape = ones([R,1])*eps;
x_rate = ones([R,1])*eps;
n_shape = 1;
n_rate = noise_variance;

% x_shape = x_shape_prior;
% x_mean_square = inv(x_precision)+x_mean*x_mean';
% x_rate = diag(x_mean_square)+x_rate_prior;

for t = 1:T
    
    
    % updata x
    D_mean_square = 1;
    for l = L:-1:1
        D_mean_square = D_mean_square.*(D_mean{l}'*D_mean{l});
    end
    % calculate x_precision
    x_precision = n_shape/n_rate*D_mean_square+diag(x_shape./x_rate);
    D_mean_all = ones(1,R);
    for l = L:-1:1
        D_mean_all = khatrirao(D_mean_all,D_mean{l});
    end
    % calculate x_mean
    x_mean = n_shape/n_rate*inv(x_precision)*D_mean_all'*vectorized_noisy_channel;
%     % calculate x_shape annd x_rate
%     x_shape = x_shape_prior;
%     x_mean_square = inv(x_precision)+x_mean*x_mean';
%     x_rate = diag(x_mean_square)+x_rate_prior;

    
    % Fast algorithm for x_shape and x_rate
    [~,index] = sort(x_rate,'descend');
    for index_r = 1:R
        r = index(index_r);
        exclusive_x_shape_x_rate = diag(x_shape./x_rate); exclusive_x_shape_x_rate(r,r) = 0;
        x_precision_exclusive = n_shape/n_rate*D_mean_square+exclusive_x_shape_x_rate;
        exclusive_vector = zeros(R,1); exclusive_vector(r) = 1;
        sl(r) = exclusive_vector'*inv(x_precision_exclusive)*exclusive_vector;
        wl2(r) = (n_shape/n_rate)^2*exclusive_vector'*inv(x_precision_exclusive)*...
            D_mean_all'*vectorized_noisy_channel*vectorized_noisy_channel'*D_mean_all*...
            inv(x_precision_exclusive)*exclusive_vector;
        if wl2(r)-sl(r)>0
            x_rate(r) = wl2(r)-sl(r);
        else
            x_rate(r) = eps;
        end
    end




    %  Update D_mean
    update_position = (x_rate~=eps); % abs(x_mean)>sqrt(eps) x_rate~=eps
    [D_mean_update_part] = KSVD_LIKE(x_mean, x_precision, x_shape, x_rate, D_mean, noisy_channel, R, L, D_size, n_shape, n_rate);
    for l = 1:L
        D_mean_temp = D_mean{l};
        D_mean_temp(:,update_position) = D_mean_update_part{l};
        D_mean{l} = D_mean_temp;
    end


    
    % update n
    n_shape = prod(cell2mat(D_size))+n_shape_prior-1;
    n_rate = vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_mean_all'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_mean_all*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(D_mean_square,[],1) + n_rate_prior;
    
    
%     n_shape = 1;
%     n_rate = noise_variance;




%         if R<Total_R
%         for l = 1:L
%             temp = randn([D_size{l},1]) + 1i*randn([D_size{l},1]);
%             D_mean_temp = D_mean{l};
%             D_mean_temp = [D_mean_temp,temp./abs(temp)];
%             D_mean{l} = D_mean_temp;
%         end
%         x_mean = [x_mean; randn([1,1]) + 1i*randn([1,1])];
%         x_shape = [x_shape;1];
%         x_rate = [x_rate;eps];
%         x_shape_prior = [x_shape_prior;1];
%         x_rate_prior = [x_rate_prior;eps];
%         x_precision = diag(blkdiag([diag(x_precision);1]));
%         R=R+1;
%         end
%         D_mean_all = ones(1,R);
%         for l = L:-1:1
%             D_mean_all = khatrirao(D_mean_all,D_mean{l});
%         end
    
    

%     % prune zero variables
%     prune_position = (x_rate==eps);
%     % x related
%     x_mean(prune_position) = [];
%     x_precision(prune_position,:) = []; x_precision(:,prune_position) = [];
%     x_shape(prune_position) = [];
%     x_rate(prune_position) = [];
%     % D related
%     for l = 1:L
%         D_mean_temp = D_mean{l};
%         D_mean_temp(:,prune_position) = [];
%         D_mean{l} = D_mean_temp;
%     end
%     R = R - sum(prune_position);


%     x_shape_prior = ones([R,1])*eps;
%     x_rate_prior = ones([R,1])*eps;
%     sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all')

end
normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


function [D_mean] = KSVD_LIKE(x_mean, x_precision, x_shape, x_rate, D_mean, noisy_channel, R, L, D_size, n_shape, n_rate)
    % Only choose nonzero elements position
    xxxx = x_mean;
    DDDD = D_mean;
    x_rate_temp=x_rate;
    prune_position = (x_rate==eps);% abs(x_mean)<sqrt(eps) x_rate==eps
    update_position = (x_rate~=eps);
    update_column = find(update_position);
    % x related
    x_mean(prune_position) = [];
    x_precision(prune_position,:) = []; x_precision(:,prune_position) = [];
    x_shape(prune_position) = [];
    x_rate(prune_position) = [];
    % D related
    for l = 1:L
        D_mean_temp = D_mean{l};
        D_mean_temp(:,prune_position) = [];
        D_mean{l} = D_mean_temp;
    end
    R = R - sum(prune_position);
    sum(prune_position);
    if sum(prune_position)==length(x_rate_temp)
        return;
    end
    
    % update D
%     [~,index] = sort(x_rate,'descend');
    for l = 1:L
        % calculate D_mean_square_exclusive
        D_mean_square_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                D_mean_square_exclusive = D_mean_square_exclusive.*(D_mean{i}'*D_mean{i});
            end
        end
        % calculate D_precision_like
        temp = diag(kron(conj(x_mean),x_mean)+reshape(inv(x_precision),[],1));
        Eigen_Decomposition_Matrix{l} = reshape(temp*reshape(conj(D_mean_square_exclusive),[],1),R,R);
        [V_matrix{l}, D_matrix{l}] = eig(Eigen_Decomposition_Matrix{l});
        sum(abs(V_matrix{l}*D_matrix{l}*V_matrix{l}'- Eigen_Decomposition_Matrix{l}),'all');
        T_matrix{l} = V_matrix{l}*sqrt(D_matrix{l});
        % calculate D_mean_exclusive
        D_mean_exclusive = ones(1,R);
        for i = L:-1:1
            if i ~= l
                D_mean_exclusive = khatrirao(D_mean_exclusive,conj(D_mean{i}));
            end
        end
%         % calculate D_mean_like
        D_mean{l} = double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean')*inv(T_matrix{l}')*inv(T_matrix{l});
%        
%         % calculate D_mean
%         Mises_Fisher{l} = n_shape/n_rate*double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean');
%         [U_matrix{l},~,V_matrix{l}] = svd(Mises_Fisher{l},'econ');
%         D_mean{l} = U_matrix{l}*V_matrix{l}';
        
        
%         D_mean_l = double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean')*inv(T_matrix{l}')*inv(T_matrix{l});
%         temp = D_mean{l};
%         temp(:,r) = D_mean_l(:,r);
%         D_mean{l}=temp;
%         
%         D_solution_temp = double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean')*inv(T_matrix{l}')*T_matrix{l}';
%         [u,~,w] = svd(D_solution_temp,'econ');
%         D_mean{l} = u*w';
%           D_mean{l} = D_mean{l}./abs(D_mean{l});
%         D_mean{l} = D_mean{l}/norm(D_mean{l},'fro');%*R*D_size{l};
%     end
    end
    
end


%% Channel estimation using CP based EM original SBL DL
function [normalized_error] = CP_EM_original_SBL_DL_original(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
fake_noise_variance = mean(abs(vectorized_noisy_channel).^2)/100;
% Parameter Initialization
D_size = {Mx,Mz,N};
for l = 1:L
    temp = randn([D_size{l},R]) + 1i*randn([D_size{l},R]);
    D_mean{l} = temp./abs(temp);
end
x_shape_prior = ones([R,1])*2;
x_rate_prior = ones([R,1])*eps;
x_precision = diag(ones([R,1]));
x_mean = randn([R,1]) + 1i*randn([R,1]);
n_shape_prior = eps;
n_rate_prior = eps;
% Initialization
for l = 1:L
    D_shape{l} = ones([R,1])*eps;
    D_rate{l} = ones([R,1])*eps;
end
x_shape = ones([R,1])*eps;
x_rate = ones([R,1])*eps;
n_shape = 1;
n_rate = eps;

for t = 1:T
    
    update_position = (x_mean~=0);
    % update D
    for l = 1:L
        % calculate D_mean_square_exclusive
        D_mean_square_exclusive = 1;
        for i = L:-1:1
            if i ~= l
                D_mean_square_exclusive = D_mean_square_exclusive.*(D_mean{i}'*D_mean{i});
            end
        end
        % calculate D_precision_like
        temp = diag(kron(conj(x_mean),x_mean)+reshape(inv(x_precision),[],1));
        Eigen_Decomposition_Matrix{l} = reshape(temp*reshape(conj(D_mean_square_exclusive),[],1),R,R);
        [V_matrix{l}, D_matrix{l}] = eig(Eigen_Decomposition_Matrix{l});
        sum(abs(V_matrix{l}*D_matrix{l}*V_matrix{l}'- Eigen_Decomposition_Matrix{l}),'all')
        T_matrix{l} = V_matrix{l}*sqrt(D_matrix{l});
        % calculate D_mean_exclusive
        D_mean_exclusive = ones(1,R);
        for i = L:-1:1
            if i ~= l
                D_mean_exclusive = khatrirao(D_mean_exclusive,conj(D_mean{i}));
            end
        end
        % calculate D_mean_like
        D_mean{l} = double(tenmat(noisy_channel,l))*D_mean_exclusive*diag(x_mean')*inv(T_matrix{l}')*inv(T_matrix{l}');
        D_mean{l} = D_mean{l}/norm(D_mean{l},'fro');
    end
    
    
    % updata x
    D_mean_square = 1;
    for l = L:-1:1
        D_mean_square = D_mean_square.*(D_mean{l}'*D_mean{l});
    end
    % calculate x_precision
    x_precision = n_shape/n_rate*D_mean_square+diag(x_shape./x_rate);
    D_mean_all = ones(1,R);
    for l = L:-1:1
        D_mean_all = khatrirao(D_mean_all,D_mean{l});
    end
    % calculate x_mean
    x_mean = n_shape/n_rate*inv(x_precision)*D_mean_all'*vectorized_noisy_channel;
    % calculate x_shape annd x_rate
    x_shape = x_shape_prior;
    x_mean_square = inv(x_precision)+x_mean*x_mean';
    x_rate = diag(x_mean_square)+x_rate_prior;
    

%     posi = D_rate{1}<1e-6; 
%     x_rate(posi) = eps;
    
%     x_shape = (1-x_shape./x_rate.*diag(inv(x_precision)));
%     x_rate = diag(x_mean*x_mean')+x_rate_prior;
    
    % update n
    n_shape = prod(cell2mat(D_size))+n_shape_prior-1;
    n_rate = vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_mean_all'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_mean_all*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(D_mean_square,[],1) + n_rate_prior;

% n_rate=(vectorized_noisy_channel-D_mean_all*x_mean)'*(vectorized_noisy_channel-D_mean_all*x_mean)+trace(D_mean_all'*D_mean_all*inv(x_precision));
% n_shape = 1;
% n_rate = noise_variance;

%     r = (1-x_shape./x_rate.*diag(inv(x_precision)));
%     n_rate = (vectorized_noisy_channel-D_mean_all*x_mean)'*(vectorized_noisy_channel-D_mean_all*x_mean) + n_rate_prior;


    % prune zero variables
    prune_position = (x_mean<eps);
    % x related
    x_mean(prune_position) = [];
    x_precision(prune_position,:) = []; x_precision(:,prune_position) = [];
    x_shape(prune_position) = [];
    x_rate(prune_position) = [];
    % D related
    for l = 1:L
        D_mean_temp = D_mean{l};
        D_mean_temp(:,prune_position) = [];
        D_mean{l} = D_mean_temp;
    end
    R = R - sum(prune_position);
    x_shape_prior = ones([R,1])*eps;
    x_rate_prior = ones([R,1])*eps;
    
end
n_rate/n_shape > noise_variance;

normalized_error = sum(abs(D_mean_all*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


%% Channel estimation using Tucker based SBL DL (JSTSP)
function [normalized_error] = Tucker_SBL_DL_JSTSP(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = Total_R;round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_varianc = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
noise_variance(1) = noise_varianc; noise_variance(2) = 1; noise_variance(3) = 1;

% Parameter Initialization
D_size = {Mx,Mz,N};
for l = 1:L
    D_covariance{l} = diag(ones(1,R));
    temp = randn([D_size{l},R]) + 1i*randn([D_size{l},R]);
    D_mean{l} = temp./abs(temp);
    D_prior{l} = ones(R,1);
    x_covariance = diag(ones(1,Total_R));
    x_mean = randn([Total_R,1]) + 1i*randn([Total_R,1]);
    x_prior{l} = ones(R,1);
end

for t = 1:T

    % E-step
    % Complexity reduction for update of x
    for l = L:-1:1
        [Q{l},A{l},Q{l}] = eig( diag(x_prior{l}.^(1/2)) * (D_size{l}*D_covariance{l}+D_mean{l}'*D_mean{l}) * diag(x_prior{l}.^(1/2)) / noise_variance(l) );
    end
    % Update x
    x_covariance_temp_1 = 1; x_covariance_temp_2 = 1; x_covariance_temp_3 = 1; x_mean_temp_1 = 1;
    for l = L:-1:1
        x_covariance_temp_1 = kron(x_covariance_temp_1,diag(x_prior{l}.^(1/2))*Q{l});
        x_covariance_temp_2 = kron(x_covariance_temp_2,diag(A{l}));
        x_covariance_temp_3 = kron(x_covariance_temp_3,Q{l}'*diag(x_prior{l}.^(1/2)));
        x_mean_temp_1 = kron(x_mean_temp_1,Q{l}'*diag(x_prior{l}.^(1/2))*D_mean{l}'/noise_variance(l));
    end
    x_covariance = x_covariance_temp_1*diag(1./(1+x_covariance_temp_2))*x_covariance_temp_3;
    x_mean = x_covariance_temp_1*diag(1./(1+x_covariance_temp_2))*x_mean_temp_1*vectorized_noisy_channel;

    % Update D
    for l = 1:L
        % Update D_covariance
        D_covariance_temp_1 = 1;
        for i = L:-1:1
            if i ~= l
                D_covariance_temp_1 = kron(D_covariance_temp_1,conj(D_size{i}*D_covariance{i}+D_mean{i}'*D_mean{i}));
            end
        end
        x_mean_tensor = reshape(x_mean,R,R,R);
        
        % calculate x related terms
        x_mean_tensor = reshape(x_mean,R,R,R); x_mean_index_trnsor = reshape(linspace(1,R^L,R^L),R,R,R);
        x_mean_mode = double(tenmat(x_mean_tensor,l)); x_mean_index_mode = double(tenmat(x_mean_index_trnsor,l));
        D_covariance_temp_2 = kron(conj(x_mean_mode),x_mean_mode) + x_covariance(sub2ind(size(x_covariance),kron(ones(size(x_mean_index_mode)),x_mean_index_mode),kron(x_mean_index_mode,ones(size(x_mean_index_mode)))));
        % D_covariance_temp_2 = kron(conj(double(tenmat(x_mean_tensor,l))),double(tenmat(x_mean_tensor,l)));
        
        D_covariance{l} = inv(reshape(D_covariance_temp_2*reshape(D_covariance_temp_1,[],1),...
            size(D_covariance{l}))/noise_variance(1) + diag(1./D_prior{l})); % SIMPLIFIED
        % Update D_mean
        D_mean_temp = 1;
        for i = L:-1:1
            if i ~= l
                D_mean_temp = kron(D_mean_temp,conj(D_mean{i}));
            end
        end
        D_mean{l} = double(tenmat(noisy_channel,l))/noise_variance(1)*D_mean_temp*double(tenmat(x_mean_tensor,l))'*D_covariance{l};
    end

    % M-step
    for l = 1:L
        % Update D_prior
        D_prior{l} = diag(D_size{l}*D_covariance{l}+D_mean{l}'*D_mean{l})/D_size{l};
        % Update x_prior
        x_prior_temp_1 = 1/(R^2);
        
        % calculate x related terms
        x_mean_tensor = reshape(x_mean,R,R,R); x_mean_index_trnsor = reshape(linspace(1,R^L,R^L),R,R,R);
        x_mean_mode = double(tenmat(x_mean_tensor,l)); x_mean_index_mode = double(tenmat(x_mean_index_trnsor,l));
        x_prior_temp_2 = x_mean_mode.*conj(x_mean_mode) + x_covariance(sub2ind(size(x_covariance),x_mean_index_mode,x_mean_index_mode));
        % x_prior_temp_2 = double(tenmat(x_mean_tensor,l)) .* conj(double(tenmat(x_mean_tensor,l))) + eps; % SIMPLIFIED
        
        x_prior_temp_3 = 1;
        for i = L:-1:l+1
            x_prior_temp_3 = kron(x_prior_temp_3,x_prior{i});
        end 
        x_prior_temp_3 = 1./x_prior_temp_3;
        x_prior_temp_4 = 1;
        for i = l-1:-1:1
            x_prior_temp_4 = kron(x_prior_temp_4,x_prior{i});
        end
        x_prior_temp_4 = 1./x_prior_temp_4;
        x_prior_temp_5 = kron(x_prior_temp_3,x_prior_temp_4);
        x_prior{l} = x_prior_temp_1*x_prior_temp_2*x_prior_temp_5;
    end
    
end

% Reconstruct D and x
D = 1;
for l = L:-1:1
    D = kron(D,D_mean{l});
end
normalized_error = sum(abs(D*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
% sum(abs(reshape(double(hosvd(tensor(noisy_channel),2,'rank',[2,2,2])),[],1)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all')
% [U,~]=cpd((noisy_channel),2);
% sum(abs(sum(khatrirao(U{3},U{2},U{1}),2)-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all')

end


%% Channel estimation using CP based full SBL SR (Dictionary is known)
function [normalized_error] = CP_FULL_SBL_SR(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
% Parameter Initialization
D_size = {Mx,Mz,N}; 
x_shape_prior = ones([R,1])*eps;
x_rate_prior = ones([R,1])*eps;
x_precision = diag(ones([R,1]));
x_mean = randn([R,1]) + 1i*randn([R,1]);
n_shape_prior = eps;
n_rate_prior = eps;
% Initialization
x_shape = x_shape_prior;
x_rate = x_rate_prior;
n_shape = n_shape_prior;
n_rate = n_rate_prior;
% Generate Dictionary
[w_theta_estimated,w_varphi_estimated,w_etatau_estimated] = MD_ESPRIT(Experiment_setting, noisy_channel, R);
[D_w_theta, D_w_varphi, D_w_etatau] = generate_sub_dictionaries(Experiment_setting, w_theta_estimated, w_varphi_estimated, w_etatau_estimated);
D_mean = khatrirao(D_w_etatau, D_w_varphi, D_w_theta);

for t = 1:T
    
    % updata x
    % calculate x_precision
    x_precision = n_shape/n_rate*(D_mean'*D_mean+diag(x_shape./x_rate));
    % calculate x_mean
    x_mean = n_shape/n_rate*inv(x_precision)*D_mean'*vectorized_noisy_channel;
    % calculate x_shape annd x_rate
    x_shape = 1+x_shape_prior;
    x_mean_square = inv(x_precision)+x_mean*x_mean';
    x_rate = n_shape/n_rate*diag(x_mean_square)+x_rate_prior;
    
    % update n
    % calculate n_shape annd n_rate
    n_shape = prod(cell2mat(D_size))*R+n_shape_prior;
    n_rate = vectorized_noisy_channel'*vectorized_noisy_channel ...
        - x_mean'*D_mean'*vectorized_noisy_channel ...
        - vectorized_noisy_channel'*D_mean*x_mean ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(D_mean'*D_mean,[],1) ...
        + (kron(x_mean.',x_mean')+reshape(inv(x_precision).',1,[]))*reshape(diag(x_shape./x_rate),[],1) + n_rate_prior;    
end
normalized_error = sum(abs(D_mean*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


%% Channel estimation using Tucker based SBL SR (JSTSP) (Dictionary is known)
function [normalized_error] = Tucker_SBL_SR_JSTSP(Experiment_setting, path_setting)
% Read setting
[~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, L, Total_R, T] = deal(Experiment_setting{:});
R = round(power(Total_R, 1/L));
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(vectorized_noisy_channel-vectorized_original_channel).^2);
% Generate Dictionary
[w_theta_estimated,w_varphi_estimated,w_etatau_estimated] = MD_ESPRIT(Experiment_setting, noisy_channel, R);
[D_w_theta, D_w_varphi, D_w_etatau] = generate_sub_dictionaries(Experiment_setting, w_theta_estimated, w_varphi_estimated, w_etatau_estimated);
D = {D_w_theta, D_w_varphi, D_w_etatau};
% Parameter Initialization
for l = 1:L
    D_mean{l} = D{l};
    x_covariance = diag(ones(1,Total_R));
    x_mean = randn([Total_R,1]) + 1i*randn([Total_R,1]);
    x_prior{l} = ones(R,1);
end
for t = 1:T

    % E-step
    % Complexity reduction for update of x
    for l = L:-1:1
        [Q{l},A{l},Q{l}] = eig( diag(x_prior{l}.^(1/2)) * (D_mean{l}'*D_mean{l}) * diag(x_prior{l}.^(1/2)) / noise_variance );
    end
    % Update x
    x_covariance_temp_1 = 1; x_covariance_temp_2 = 1; x_covariance_temp_3 = 1; x_mean_temp_1 = 1;
    for l = L:-1:1
        x_covariance_temp_1 = kron(x_covariance_temp_1,diag(x_prior{l}.^(1/2))*Q{l});
        x_covariance_temp_2 = kron(x_covariance_temp_2,diag(A{l}));
        x_covariance_temp_3 = kron(x_covariance_temp_3,Q{l}'*diag(x_prior{l}.^(1/2)));
        x_mean_temp_1 = kron(x_mean_temp_1,Q{l}'*diag(x_prior{l}.^(1/2))*D_mean{l}'/noise_variance);
    end
    x_covariance = x_covariance_temp_1*diag(1./(1+x_covariance_temp_2))*x_covariance_temp_3;
    x_mean = x_covariance_temp_1*diag(1./(1+x_covariance_temp_2))*x_mean_temp_1*vectorized_noisy_channel;
    x_mean_tensor = reshape(x_mean,R,R,R);

    % M-step
    for l = 1:L
        % Update x_prior
        x_prior_temp_1 = 1/(R^2);
        x_prior_temp_2 = double(tenmat(x_mean_tensor,l)) .* conj(double(tenmat(x_mean_tensor,l))) + eps; % SIMPLIFIED
        x_prior_temp_3 = 1;
        for i = L:-1:l+1
            x_prior_temp_3 = kron(x_prior_temp_3,x_prior{i});
        end 
        x_prior_temp_3 = 1./x_prior_temp_3;
        x_prior_temp_4 = 1;
        for i = l-1:-1:1
            x_prior_temp_4 = kron(x_prior_temp_4,x_prior{i});
        end
        x_prior_temp_4 = 1./x_prior_temp_4;
        x_prior_temp_5 = kron(x_prior_temp_3,x_prior_temp_4);
        x_prior{l} = x_prior_temp_1*x_prior_temp_2*x_prior_temp_5;
    end
    
end

% Reconstruct D and x
D = 1;
for l = L:-1:1
    D = kron(D,D_mean{l});
end
normalized_error = sum(abs(D*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


%% Channel estimation using Parameterized SBL
function [normalized_error] = Parameterized_SBL(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, ~, R, T] = deal(Experiment_setting{:});
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting);
% Parameter Initialization
D_size = {Mx,Mz,N}; 

n_prior = 1;
n_shape = eps;
n_rate = eps;

x_shape = ones(R,1)*2; % DONOT CHANGE
x_rate = ones(R,1)*eps; % DONOT CHANGE
x_prior = ones(R,1)/2;% SET UP TO x real precision

[w_theta_estimated,w_varphi_estimated,w_etatau_estimated] = MD_ESPRIT(Experiment_setting, noisy_channel, R);
for t = 1:T
% Reconstruct sub dictionaries
[D_w_theta, D_w_varphi, D_w_etatau] = generate_sub_dictionaries(Experiment_setting, w_theta_estimated, w_varphi_estimated, w_etatau_estimated);
D_estimated = khatrirao(D_w_etatau, D_w_varphi, D_w_theta);
[D_w_theta_first_derivative, D_w_varphi_first_derivative, D_w_etatau_first_derivative] = generate_sub_dictionaries_first_derivative(Experiment_setting, w_theta_estimated, w_varphi_estimated, w_etatau_estimated);
[D_w_theta_second_derivative, D_w_varphi_second_derivative, D_w_etatau_second_derivative] = generate_sub_dictionaries_second_derivative(Experiment_setting, w_theta_estimated, w_varphi_estimated, w_etatau_estimated);

% Update x
x_precision = n_prior*(diag(x_prior)+D_estimated'*D_estimated);
x_mean = inv(x_precision)*D_estimated'*n_prior*vectorized_noisy_channel;

% Update n_prior (i.e., noise precision)
inverse_marginal_covariance = Woodbury(eye(prod(cell2mat(D_size))), D_estimated, diag(1./x_prior), D_estimated');
n_prior = real((prod(cell2mat(D_size))+n_shape-1)/(n_rate+vectorized_noisy_channel'*inverse_marginal_covariance*vectorized_noisy_channel));

% Update each path parameter
for r = 1:R
    % update x_prior
    x_prior_exclusive = 1./x_prior;
    x_prior_exclusive(r) = 0;
    inverse_marginal_covariance_exclusive = Woodbury(eye(prod(cell2mat(D_size))), D_estimated, diag(x_prior_exclusive+eps), D_estimated');
    s_r = real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*D_estimated(:,r));
    q_r = D_estimated(:,r)'*inverse_marginal_covariance_exclusive*vectorized_noisy_channel;
    total_candidate = real(roots([-x_rate(r), x_shape(r)-1-2*x_rate(r)*s_r, -s_r^2*x_rate(r)+s_r-n_prior*abs(q_r)^2+2*(x_shape(r)-1)*s_r, (x_shape(r)-1+1)*s_r^2]));
    positive_candidate = total_candidate(total_candidate>0);
    [~,max_position] = max( -log(1+s_r./positive_candidate) + n_prior*abs(q_r)^2./(positive_candidate+s_r) ...
        +(x_shape(r)-1)*log(positive_candidate) -x_rate(r)*positive_candidate);
    x_prior(r) = positive_candidate(max_position);
    
    % update w_theta
    % w_theta first derivative
    w_theta_first_derivative_temp = khatrirao(D_w_etatau(:,r), D_w_varphi(:,r), D_w_theta_first_derivative(:,r));
    w_theta_second_derivative_temp = khatrirao(D_w_etatau(:,r), D_w_varphi(:,r), D_w_theta_second_derivative(:,r));
    s_r_first_derivative = 2*real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*w_theta_first_derivative_temp);
    q_r_first_derivative = 2*real(q_r*vectorized_noisy_channel'*inverse_marginal_covariance_exclusive'*w_theta_first_derivative_temp);
    first_order_w_theta = n_prior/(x_prior(r)+s_r)*q_r_first_derivative - (1/(x_prior(r)+s_r)+n_prior*abs(q_r)^2/(x_prior(r)+s_r)^2)*s_r_first_derivative;
    % w_theta second derivative
    s_r_second_derivative = 2*real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*w_theta_second_derivative_temp + ...
        w_theta_first_derivative_temp'*inverse_marginal_covariance_exclusive*w_theta_first_derivative_temp);
    q_r_second_derivative = 2*real(vectorized_noisy_channel'*inverse_marginal_covariance_exclusive*...
        (w_theta_first_derivative_temp*w_theta_first_derivative_temp'+w_theta_second_derivative_temp*D_estimated(:,r)')*...
        inverse_marginal_covariance_exclusive*vectorized_noisy_channel);
    second_order_w_theta = 1/(x_prior(r)+s_r)*(n_prior*q_r_second_derivative-s_r_second_derivative) ...
        + 2*n_prior*abs(q_r)^2/(x_prior(r)+s_r)^3*s_r_first_derivative^2 ...
        + 1/(x_prior(r)+s_r)^2 * (s_r_first_derivative^2 - n_prior*abs(q_r)^2*s_r_second_derivative - n_prior*s_r_first_derivative*q_r_first_derivative);
    w_theta_estimated(r) = w_theta_estimated(r) - real(first_order_w_theta)/real(second_order_w_theta);
    
    % update w_varphi
    % w_varphi first derivative
    w_varphi_first_derivative_temp = khatrirao(D_w_etatau(:,r), D_w_varphi_first_derivative(:,r), D_w_theta(:,r));
    w_varphi_second_derivative_temp = khatrirao(D_w_etatau(:,r), D_w_varphi_second_derivative(:,r), D_w_theta(:,r));
    s_r_first_derivative = 2*real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*w_varphi_first_derivative_temp);
    q_r_first_derivative = 2*real(q_r*vectorized_noisy_channel'*inverse_marginal_covariance_exclusive'*w_varphi_first_derivative_temp);
    first_order_w_varphi = n_prior/(x_prior(r)+s_r)*q_r_first_derivative - (1/(x_prior(r)+s_r)+n_prior*abs(q_r)^2/(x_prior(r)+s_r)^2)*s_r_first_derivative;
    % w_varphi second derivative
    s_r_second_derivative = 2*real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*w_varphi_second_derivative_temp + ...
        w_varphi_first_derivative_temp'*inverse_marginal_covariance_exclusive*w_varphi_first_derivative_temp);
    q_r_second_derivative = 2*real(vectorized_noisy_channel'*inverse_marginal_covariance_exclusive*...
        (w_varphi_first_derivative_temp*w_varphi_first_derivative_temp'+w_varphi_second_derivative_temp*D_estimated(:,r)')*...
        inverse_marginal_covariance_exclusive*vectorized_noisy_channel);
    second_order_w_varphi = 1/(x_prior(r)+s_r)*(n_prior*q_r_second_derivative-s_r_second_derivative) ...
        + 2*n_prior*abs(q_r)^2/(x_prior(r)+s_r)^3*s_r_first_derivative^2 ...
        + 1/(x_prior(r)+s_r)^2 * (s_r_first_derivative^2 - n_prior*abs(q_r)^2*s_r_second_derivative - n_prior*s_r_first_derivative*q_r_first_derivative);
    w_varphi_estimated(r) = w_varphi_estimated(r) - real(first_order_w_varphi)/real(second_order_w_varphi);
    
    % update w_etautau
    % w_etatau first derivative
    w_etatau_first_derivative_temp = khatrirao(D_w_etatau_first_derivative(:,r), D_w_varphi(:,r), D_w_theta(:,r));
    w_etatau_second_derivative_temp = khatrirao(D_w_etatau_second_derivative(:,r), D_w_varphi(:,r), D_w_theta(:,r));
    s_r_first_derivative = 2*real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*w_etatau_first_derivative_temp);
    q_r_first_derivative = 2*real(q_r*vectorized_noisy_channel'*inverse_marginal_covariance_exclusive'*w_etatau_first_derivative_temp);
    first_order_w_etatau = n_prior/(x_prior(r)+s_r)*q_r_first_derivative - (1/(x_prior(r)+s_r)+n_prior*abs(q_r)^2/(x_prior(r)+s_r)^2)*s_r_first_derivative;
    % w_etatau second derivative
    s_r_second_derivative = 2*real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*w_etatau_second_derivative_temp + ...
        w_etatau_first_derivative_temp'*inverse_marginal_covariance_exclusive*w_etatau_first_derivative_temp);
    q_r_second_derivative = 2*real(vectorized_noisy_channel'*inverse_marginal_covariance_exclusive*...
        (w_etatau_first_derivative_temp*w_etatau_first_derivative_temp'+w_etatau_second_derivative_temp*D_estimated(:,r)')*...
        inverse_marginal_covariance_exclusive*vectorized_noisy_channel);
    second_order_w_etatau = 1/(x_prior(r)+s_r)*(n_prior*q_r_second_derivative-s_r_second_derivative) ...
        + 2*n_prior*abs(q_r)^2/(x_prior(r)+s_r)^3*s_r_first_derivative^2 ...
        + 1/(x_prior(r)+s_r)^2 * (s_r_first_derivative^2 - n_prior*abs(q_r)^2*s_r_second_derivative - n_prior*s_r_first_derivative*q_r_first_derivative);
    w_etatau_estimated(r) = w_etatau_estimated(r) - real(first_order_w_etatau)/real(second_order_w_etatau);
end
end
normalized_error = sum(abs(D_estimated*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


%% Channel estimation using Truncated Parameterized SBL
function [normalized_error] = Truncated_Parameterized_SBL(Experiment_setting, path_setting)
% Read setting
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, ~, R, T] = deal(Experiment_setting{:});
Pilot_number = 16;
Indices = [linspace(1,Pilot_number/2,Pilot_number/2),linspace(N-Pilot_number/2+1,N,Pilot_number/2)];
% Generate pilot channels
[~, noisy_channel, vectorized_original_channel, bbb] = generate_channel(Experiment_setting, path_setting);
noise_variance = mean(abs(bbb-vectorized_original_channel).^2);
% Parameter Initialization
D_size = {Mx,Mz,Pilot_number}; 

n_prior = 1;
n_shape_prior = eps;
n_rate_prior = eps;

x_shape = ones(R,1)*1;
x_rate = ones(R,1)*0;
x_prior = ones(R,1);

[w_theta_estimated,w_varphi_estimated,w_etatau_estimated] = MD_ESPRIT(Experiment_setting, noisy_channel, R);
% [alpha, w_theta_estimated, w_varphi_estimated, w_etatau_estimated] = deal(path_setting{:});
% Truncated channel
noisy_channel = noisy_channel(:,:,Indices);
vectorized_noisy_channel = reshape(noisy_channel,[],1);
for t = 1:T
% Reconstruct sub dictionaries
[D_w_theta, D_w_varphi, ~] = generate_sub_dictionaries(Experiment_setting, w_theta_estimated, w_varphi_estimated, w_etatau_estimated);
% Truncated sub dictionary
D_w_etatau = exp(-1i*(Indices-1).'*w_etatau_estimated); 
D_estimated = khatrirao(D_w_etatau, D_w_varphi, D_w_theta);
[D_w_theta_first_derivative, D_w_varphi_first_derivative, ~] = generate_sub_dictionaries_first_derivative(Experiment_setting, w_theta_estimated, w_varphi_estimated, w_etatau_estimated);
% Truncated first derivative
D_w_etatau_first_derivative = repmat(-1i*(Indices-1).', 1, R).*exp(-1i*(Indices-1).'*w_etatau_estimated);
[D_w_theta_second_derivative, D_w_varphi_second_derivative, ~] = generate_sub_dictionaries_second_derivative(Experiment_setting, w_theta_estimated, w_varphi_estimated, w_etatau_estimated);
% Truncated second derivative
D_w_etatau_second_derivative = repmat(-1i*(Indices-1).', 1, R).*repmat(-1i*(Indices-1).', 1, R).*exp(-1i*(Indices-1).'*w_etatau_estimated);


% Update x
x_precision = n_prior*D_estimated'*D_estimated+diag(x_prior);
x_mean = n_prior*inv(x_precision)*D_estimated'*vectorized_noisy_channel;

% Update n_prior (i.e., noise precision)
n_prior = (prod(cell2mat(D_size))+n_shape_prior-1)/(sum(abs(vectorized_noisy_channel-D_estimated*x_mean).^2,'all')+trace(inv(x_precision)*D_estimated'*D_estimated) + n_rate_prior);

% Update each path parameter
for r = 1:R
    % update x_prior
    x_prior_exclusive = 1./x_prior;
    x_prior_exclusive(r) = 0;
    inverse_marginal_covariance_exclusive = Woodbury(n_prior, n_prior*eye(prod(cell2mat(D_size))), D_estimated, diag(x_prior_exclusive), D_estimated');
    s_r = D_estimated(:,r)'*inverse_marginal_covariance_exclusive*D_estimated(:,r);
    q_r = D_estimated(:,r)'*inverse_marginal_covariance_exclusive*vectorized_noisy_channel;
    total_candidate = roots([-x_rate(r), x_shape(r)-1-2*x_rate(r)*s_r, 2*x_shape(r)*s_r-s_r-x_rate(r)*s_r^2-abs(q_r)^2, x_shape(r)*s_r^2]);
    positive_candidate = total_candidate(total_candidate>0);
    if isempty(positive_candidate)
        positive_candidate = [positive_candidate,1/eps];
    end
    [~,max_position] = max( -log(positive_candidate+s_r) + abs(q_r)^2./(positive_candidate+s_r) ...
        + x_shape(r)*log(positive_candidate) - x_rate(r)*positive_candidate);
    x_prior(r) = positive_candidate(max_position);

    % update w_theta
    % w_theta first derivative
    w_theta_first_derivative_temp = khatrirao(D_w_etatau(:,r), D_w_varphi(:,r), D_w_theta_first_derivative(:,r));
    w_theta_second_derivative_temp = khatrirao(D_w_etatau(:,r), D_w_varphi(:,r), D_w_theta_second_derivative(:,r));
    s_r_first_derivative = 2*real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*w_theta_first_derivative_temp);
    q_r_first_derivative = 2*real(q_r*vectorized_noisy_channel'*inverse_marginal_covariance_exclusive'*w_theta_first_derivative_temp);
    first_order_w_theta = -(1/(x_prior(r)+s_r)+abs(q_r)^2/(x_prior(r)+s_r)^2)*s_r_first_derivative + 1/(x_prior(r)+s_r)*q_r_first_derivative;
    % w_theta second derivative
    s_r_second_derivative = 2*real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*w_theta_second_derivative_temp + ...
        w_theta_first_derivative_temp'*inverse_marginal_covariance_exclusive*w_theta_first_derivative_temp);
    q_r_second_derivative = 2*real(vectorized_noisy_channel'*inverse_marginal_covariance_exclusive*...
        (w_theta_first_derivative_temp*w_theta_first_derivative_temp'+w_theta_second_derivative_temp*D_estimated(:,r)')*...
        inverse_marginal_covariance_exclusive*vectorized_noisy_channel);
    second_order_w_theta = -(1/(x_prior(r)+s_r)+abs(q_r)^2/(x_prior(r)+s_r)^2)*s_r_second_derivative + ...
        (1/(x_prior(r)+s_r)^2+2*abs(q_r)^2/(x_prior(r)+s_r)^3)*(s_r_first_derivative)^2 - ...
        2/(x_prior(r)+s_r)^2*s_r_first_derivative*q_r_first_derivative + ...
        1/(x_prior(r)+s_r)*q_r_second_derivative;
    w_theta_estimated(r) = w_theta_estimated(r) - real(first_order_w_theta)/real(second_order_w_theta);
    
    % update w_varphi
    % w_varphi first derivative
    w_varphi_first_derivative_temp = khatrirao(D_w_etatau(:,r), D_w_varphi_first_derivative(:,r), D_w_theta(:,r));
    w_varphi_second_derivative_temp = khatrirao(D_w_etatau(:,r), D_w_varphi_second_derivative(:,r), D_w_theta(:,r));
    s_r_first_derivative = 2*real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*w_varphi_first_derivative_temp);
    q_r_first_derivative = 2*real(q_r*vectorized_noisy_channel'*inverse_marginal_covariance_exclusive'*w_varphi_first_derivative_temp);
    first_order_w_varphi = -(1/(x_prior(r)+s_r)+abs(q_r)^2/(x_prior(r)+s_r)^2)*s_r_first_derivative + 1/(x_prior(r)+s_r)*q_r_first_derivative;
    % w_varphi second derivative
    s_r_second_derivative = 2*real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*w_varphi_second_derivative_temp + ...
        w_varphi_first_derivative_temp'*inverse_marginal_covariance_exclusive*w_varphi_first_derivative_temp);
    q_r_second_derivative = 2*real(vectorized_noisy_channel'*inverse_marginal_covariance_exclusive*...
        (w_varphi_first_derivative_temp*w_varphi_first_derivative_temp'+w_varphi_second_derivative_temp*D_estimated(:,r)')*...
        inverse_marginal_covariance_exclusive*vectorized_noisy_channel);
    second_order_w_varphi = -(1/(x_prior(r)+s_r)+abs(q_r)^2/(x_prior(r)+s_r)^2)*s_r_second_derivative + ...
        (1/(x_prior(r)+s_r)^2+2*abs(q_r)^2/(x_prior(r)+s_r)^3)*(s_r_first_derivative)^2 - ...
        2/(x_prior(r)+s_r)^2*s_r_first_derivative*q_r_first_derivative + ...
        1/(x_prior(r)+s_r)*q_r_second_derivative;
    w_varphi_estimated(r) = w_varphi_estimated(r) - real(first_order_w_varphi)/real(second_order_w_varphi);
    
    % update w_etautau
    % w_etatau first derivative
    w_etatau_first_derivative_temp = khatrirao(D_w_etatau_first_derivative(:,r), D_w_varphi(:,r), D_w_theta(:,r));
    w_etatau_second_derivative_temp = khatrirao(D_w_etatau_second_derivative(:,r), D_w_varphi(:,r), D_w_theta(:,r));
    s_r_first_derivative = 2*real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*w_etatau_first_derivative_temp);
    q_r_first_derivative = 2*real(q_r*vectorized_noisy_channel'*inverse_marginal_covariance_exclusive'*w_etatau_first_derivative_temp);
    first_order_w_etatau = -(1/(x_prior(r)+s_r)+abs(q_r)^2/(x_prior(r)+s_r)^2)*s_r_first_derivative + 1/(x_prior(r)+s_r)*q_r_first_derivative;
    % w_etatau second derivative
    s_r_second_derivative = 2*real(D_estimated(:,r)'*inverse_marginal_covariance_exclusive*w_etatau_second_derivative_temp + ...
        w_etatau_first_derivative_temp'*inverse_marginal_covariance_exclusive*w_etatau_first_derivative_temp);
    q_r_second_derivative = 2*real(vectorized_noisy_channel'*inverse_marginal_covariance_exclusive*...
        (w_etatau_first_derivative_temp*w_etatau_first_derivative_temp'+w_etatau_second_derivative_temp*D_estimated(:,r)')*...
        inverse_marginal_covariance_exclusive*vectorized_noisy_channel);
    second_order_w_etatau = -(1/(x_prior(r)+s_r)+abs(q_r)^2/(x_prior(r)+s_r)^2)*s_r_second_derivative + ...
        (1/(x_prior(r)+s_r)^2+2*abs(q_r)^2/(x_prior(r)+s_r)^3)*(s_r_first_derivative)^2 - ...
        2/(x_prior(r)+s_r)^2*s_r_first_derivative*q_r_first_derivative + ...
        1/(x_prior(r)+s_r)*q_r_second_derivative;
    w_etatau_estimated(r) = w_etatau_estimated(r) - real(first_order_w_etatau)/real(second_order_w_etatau);
end
end
[D_w_theta, D_w_varphi, D_w_etatau] = generate_sub_dictionaries(Experiment_setting, w_theta_estimated, w_varphi_estimated, w_etatau_estimated);
D_estimated = khatrirao(D_w_etatau, D_w_varphi, D_w_theta);
normalized_error = sum(abs(D_estimated*x_mean-vectorized_original_channel).^2,'all')/sum(abs(vectorized_original_channel).^2,'all');
end


%% Multidimensional ESPRIT
function [w_theta_estimated,w_varphi_estimated,w_etatau_estimated] = MD_ESPRIT(Experiment_setting, noisy_channel, R)
% Read setting
[Mx, Mz, Nf, ~, ~, ~, ~, ~, ~, ~, ~, N, ~, ~] = deal(Experiment_setting{:});
% Step 1
L = [Mx/2,Mz/2,Nf/2];
K = [Mx,Mz,Nf]-L+1;
% Step 2
H = [];
for l=1:K(1)
    for m=1:K(2)
        for n=1:K(3)
            i = [l,m,n];
            H = [H,reshape(noisy_channel(i(1):i(1)+L(1)-1,i(2):i(2)+L(2)-1,i(3):i(3)+L(3)-1),[],1)];
        end
    end
end
% Step 3
[U,~,~] = svd(H);
U_s = U(:,1:R);
% Step 4
I_temp = {eye(L(1)),eye(L(2)),eye(L(3))}; n_I = {}; I_n = {};
for n=1:N
    % I_n
    I_temp_temp = I_temp;
    temp = I_temp{n};
    temp(end,:) = [];
    I_temp_temp{n} = temp;
    I_n{n} = kron(kron(I_temp_temp{3},I_temp_temp{2}),I_temp_temp{1}); 
    % n_I
    I_temp_temp = I_temp;
    temp = I_temp{n};
    temp(1,:) = [];
    I_temp_temp{n} = temp;
    n_I{n} = kron(kron(I_temp_temp{3},I_temp_temp{2}),I_temp_temp{1}); 
end
F = {};
for n=1:N
    F{n} = pinv(I_n{n}*U_s)*(n_I{n}*U_s);
end
% Step 5
coefficient = randn(1,N);
K = 0;
for n=1:N
    K = K+F{n}*coefficient(n);
end
% Step 6
[T,~]=eig(K);
% Step 7
D = {};
for n=1:N
    D{n} = inv(T)*F{n}*T;
end
% Step 8
w_theta_estimated = -angle(diag(D{1}))';
w_varphi_estimated = -angle(diag(D{2}))';
w_etatau_estimated = -angle(diag(D{3}))';
end


%% Generate original and noisy channel
function [original_channel, noisy_channel, vectorized_original_channel, vectorized_noisy_channel] = generate_channel(Experiment_setting, path_setting)
% Read settings
[Mx, Mz, N, SNR, ~, ~, ~, ~, ~, ~, P, ~, ~, ~] = deal(Experiment_setting{:});

beta = 0.9;
varying(1,:) = randn(1,P) + 0*randn(1,P);
for i = 2:N*100
    varying(i,:) = 1 + beta.*varying(i-1,:) + sqrt(1-beta^2)*(randn(1,P) + 0*randn(1,P));
end
varying = varying(end-N+1:end,:);

% plot(varying)

varying1(1,:) = randn(1,P) + 0*randn(1,P);
for i = 2:Mx*100
    varying1(i,:) = beta.*varying1(i-1,:) + sqrt(1-beta^2)*(randn(1,P) + 0*randn(1,P));
end
varying1 = varying1(end-Mx+1:end,:);

[alpha, w_theta, w_varphi, w_etatau] = deal(path_setting{:});
exp(-1i*linspace(0,N-1,N).'*w_etatau);ones(N,P); Random=rand(N,P)+1i*rand(N,P);
% Tensor based channel generation (UPA)
Sigma = rand(Mx);
spatial_steering_vector_x = repmat(reshape(randn([Mx,P]) + 1i*randn([Mx,P]),Mx,1,1,P),1,Mz,N,1);
Sigma = rand(Mz);
spatial_steering_vector_z = repmat(reshape(randn([Mz,P]) + 1i*randn([Mz,P]),1,Mz,1,P),Mx,1,N,1);
spatial_steering_vector = spatial_steering_vector_x.*spatial_steering_vector_z;
Sigma = randn(N);
frequency_steering_vector = repmat(reshape(0*mvnrnd(zeros([N,1]),Sigma*Sigma',P).'...
    +0*mvnrnd(zeros([N,1]),Sigma*Sigma',P).'+1*varying,1,1,N,P),Mx,Mz,1,1);
path_gain = repmat(reshape(alpha,1,1,1,P),Mx,Mz,N,1);

% original_channel = 0;
% for p = 1:P
%     original_channel = original_channel + exp(-1i*linspace(0,Mx-1,Mx).'*w_theta(p)) *...
%         exp(-1i*linspace(0,Mz-1,Mz).'*w_varphi);
% end

% for i = 1:Mx
%     for j = 1:Mz
%         for k = 1:N
%             temp = 0;
%             for p = 1:P
% %                 temp = temp + alpha(p)*exp(-1i*(i-1)*w_theta(p)) * exp(-1i*(j-1)*w_varphi(p)) ...
% %                     * exp(-1i*(k-1)*w_etatau(p));
%                 temp = temp + alpha(p)*randn * randn ...
%                     * varying(k,p);
%             end
%             original_channel(i,j,k) = temp;
%         end
%     end
% end

% Create random 50 x 40 x 30 tensor with 5 x 4 x 3 core
info = create_problem('Type','Tucker','Num_Factors',[P P P],'Size',[Mx Mz N]);
X = info.Data;double(X);

original_channel = sum((path_gain.*spatial_steering_vector.*frequency_steering_vector),4);
noisy_channel = awgn(original_channel, SNR, 'measured');

% Khatri–Rao based channel generation (UPA)
spatial_steering_matrix_x = exp(-1i*linspace(0,Mx-1,Mx).'*w_theta);
spatial_steering_matrix_z = exp(-1i*linspace(0,Mz-1,Mz).'*w_varphi);
frequency_steering_matrix = varying;
dictionary = khatrirao(frequency_steering_matrix, spatial_steering_matrix_z, spatial_steering_matrix_x);
sparse_coefficient = alpha.';
vectorized_original_channel = reshape(original_channel,[],1);dictionary*sparse_coefficient;
% vectorized_noisy_channel = awgn(vectorized_original_channel, SNR, 'measured');
vectorized_noisy_channel = reshape(noisy_channel,[],1);

end


%% Generate sub dictionaries
function [D_w_theta, D_w_varphi, D_w_etatau] = generate_sub_dictionaries(Experiment_setting, w_theta_estimated, w_varphi_estimated, w_etatau_estimated)
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = deal(Experiment_setting{:});
D_w_theta = exp(-1i*linspace(0,Mx-1,Mx).'*w_theta_estimated);
D_w_varphi = exp(-1i*linspace(0,Mz-1,Mz).'*w_varphi_estimated);
D_w_etatau = exp(-1i*linspace(0,N-1,N).'*w_etatau_estimated);
end


%% Generate sub dictionaries first derivative
function [D_w_theta_first_derivative, D_w_varphi_first_derivative, D_w_etatau_first_derivative] = generate_sub_dictionaries_first_derivative(Experiment_setting, w_theta_estimated, w_varphi_estimated, w_etatau_estimated)
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, ~, R, ~] = deal(Experiment_setting{:});
D_w_theta_first_derivative = repmat(-1i*linspace(0,Mx-1,Mx).', 1, R).*exp(-1i*linspace(0,Mx-1,Mx).'*w_theta_estimated);
D_w_varphi_first_derivative = repmat(-1i*linspace(0,Mz-1,Mz).', 1, R).*exp(-1i*linspace(0,Mz-1,Mz).'*w_varphi_estimated);
D_w_etatau_first_derivative = repmat(-1i*linspace(0,N-1,N).', 1, R).*exp(-1i*linspace(0,N-1,N).'*w_etatau_estimated);
end


%% Generate sub dictionaries second derivative
function [D_w_theta_second_derivative, D_w_varphi_second_derivative, D_w_etatau_second_derivative] = generate_sub_dictionaries_second_derivative(Experiment_setting, w_theta_estimated, w_varphi_estimated, w_etatau_estimated)
[Mx, Mz, N, ~, ~, ~, ~, ~, ~, ~, ~, ~, R, ~] = deal(Experiment_setting{:});
D_w_theta_second_derivative = repmat(-1i*linspace(0,Mx-1,Mx).', 1, R).*repmat(-1i*linspace(0,Mx-1,Mx).', 1, R).*exp(-1i*linspace(0,Mx-1,Mx).'*w_theta_estimated);
D_w_varphi_second_derivative = repmat(-1i*linspace(0,Mz-1,Mz).', 1, R).*repmat(-1i*linspace(0,Mz-1,Mz).', 1, R).*exp(-1i*linspace(0,Mz-1,Mz).'*w_varphi_estimated);
D_w_etatau_second_derivative = repmat(-1i*linspace(0,N-1,N).', 1, R).*repmat(-1i*linspace(0,N-1,N).', 1, R).*exp(-1i*linspace(0,N-1,N).'*w_etatau_estimated);
end


%% Woodbury matrix identity
function [inverse_matrix] = Woodbury(A_diag, A, U, B, V)
inverse_matrix = A-A_diag*U*inv(eye(size(B))+A_diag*B*V*U)*B*V*A_diag;
end