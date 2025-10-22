import numpy as np
def feature_block_normalization(dat):
    
    input_features = []
    transcriptions = []
    frame_lens = []
    blocks = []
    
    n_trials = dat['sentenceText'].shape[0]
    for i in range(n_trials):    
            #get time series of TX and spike power for this trial
            #first 128 columns = area 6v only
        
            features = np.concatenate([dat['tx1'][0,i][:,0:128], dat['spikePow'][0,i][:,0:128]], axis=1)
            input_features.append(features)
    
            sentence = dat['sentenceText'][i].strip()
            transcriptions.append(sentence)

            sentence_len = features.shape[0]
            frame_lens.append(sentence_len)

            blockNums = np.squeeze(dat['blockIdx'])
            blockList = np.unique(blockNums)
            
    for b in range(len(blockList)):
                sentIdx = np.argwhere(blockNums==blockList[b])
                sentIdx = sentIdx[:,0].astype(np.int32)
                blocks.append(sentIdx)
    for b in range(len(blocks)):
        feats = np.concatenate(input_features[blocks[b][0]:(blocks[b][-1]+1)], axis=0)
        feats_mean = np.mean(feats, axis=0, keepdims=True)
        feats_std = np.std(feats, axis=0, keepdims=True)
        for i in blocks[b]:
            input_features[i] = (input_features[i] - feats_mean) / (feats_std + 1e-8)
            
    session_data = {
            'inputFeatures': input_features,
            'transcriptions': transcriptions,
            'frameLens': frame_lens
        }
    return session_data