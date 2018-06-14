for datasize = 1000000:1000000
addpath('yael_v403_20/matlab');

cfg = config_oxford();

% Parameters
nbits = 128;        % dimension of binary signatures
ht = 0;	            % similarity threshold, possible values [0,nbits/2]
alpha = 3.0;        % parameter of the selective function sign(u)*u.^alpha
k = 2^16;           % codebook size
ma = 5;             % multiple assigned visual words		
part = 10;

docluster = false;  % compute codebook/used a pre-computed one
compute_vw = true; % compute visual words for test set/load pre-computed ones


% -----------------------------------
% Query inverted file
% -----------------------------------

fivf_name = sprintf('yael_v438/data/ivf_flickr1M_%dK', datasize/1000);
% Load ivf
fprintf ('* Load the inverted file from %s\n', fivf_name);
ivfhe = yael_ivf_he (fivf_name);
load (sprintf ('%s_other.mat', fivf_name), 'scoremap', 'listw', 'normf');


idx = [1:-2/ivfhe.nbits:-1];
scoremap = single(sign (idx) .* abs(idx).^1.0);
scoremap = (scoremap+1)/2;
ivfhe.scoremap = scoremap;


ivfhe.listw = listw;
ivfhe.normf = normf;

% Load ground truth structure for Oxford5k
load (cfg.gnd_fname);
clear vtrain_mean

% Load test descriptors and number of features per image
fprintf ('* Loading and post-processing database descriptors\n');
load('flickr_mean.mat', 'vtrain_mean');
load(cfg.test_sift_fname);
load(cfg.test_nf_fname);
vtest = desc_postprocess (vtest, vtrain_mean);

cs = [1 cumsum( double (nftest)) + 1];
load('flickr_vladsift_thres_qidx.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dis_d = exp(-0.5*dis_d);
thres = ceil(128*0.6*dis_d);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf ('* Perform queries\n');
ntime = zeros(1, length(qidx));

% Query using 55 predefined bounding boxes on oxford images
for q=1:numel(qidx)
  
    if nftest(qidx(q)) ~= 0
  fprintf ('* Loading and postprocessing query descriptors\n');	
  % Descriptors of q-th image
  dquery = vtest (:, cs(qidx(q)):cs(qidx(q)+1)-1);
  cqidx = 1:1:size(dquery, 2);
  dquery = dquery (:, cqidx);
 
  % Compute visual words for test descriptors
  tic;
  fname = sprintf('quantization/query_%06d.mat', q);
  if(~exist(fname, 'file'))
      [vquery, ~] = ivfhe.quantizer (ivfhe.quantizer_params, dquery, ma);
      save(fname, 'vquery');
  else
      load(fname, 'vquery');
  end
%   [vquery, ~] = ivfhe.quantizer (ivfhe.quantizer_params, dquery, ma);
  fprintf ('* Computed visual words for query descriptors in %.3f seconds\n', toc);		
  
  vquery = reshape (vquery', [1 ma * numel(cqidx)]);
  dquery = repmat (dquery, 1, ma);
  nquery = size(dquery, 2);
 							
  % Descriptor aggregation per visual word
  [vquery, dquery, nquery] = aggregate_all (vquery, dquery, nquery);
  
  % Query ivf structure and collect matches
  ht = thres(:, q);
  tic;
  [matches, sim] = ivfhe.queryw (ivfhe, int32(1:nquery), dquery, ht , vquery);
  ntime(q) = toc;
%   [matches, sim] = ivfhe.queryw (ivfhe, int32(1:nquery), dquery, ht + nbits / 2, vquery);
  fprintf ('* Performed query %d in %.3f seconds\n', q, toc);

  % Compute final similarity score per image and rank
  score = accumarray (matches (2,:)', sim, [datasize+1491 1]) ./ ivfhe.normf';
  scores(:, q) = score;
  [~, ranks(:, q)] = sort (score, 'descend');
    end
end

save(fullfile(sprintf('query_flickr%dK_holidays_dth_dta20_vlad.mat', datasize/1000)), 'ranks', 'scores', 'ntime');
yael_ivf ('free');
clear

end