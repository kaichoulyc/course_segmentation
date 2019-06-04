import numpy as np

EPS = 1e-8

def compute_mask_iou(true_batch, predicted_batch):
        '''Computes iou over all batch masks.
        
        
        Parameters
        ----------
        true_batch : numpy.ndarray
            True masks batch. Format BxHxW.
        predicted_batch : numpy.ndarray
            Predicted masks batch. Format BxHxW.
        
        Returns
        -------
        iou_array : numpy.ndarray
            IOU for all masks.
        
        '''
        
        if not isinstance(predicted_batch, np.ndarray):
            raise Exception('Variable `predicted_batch` has type ' + str(type(predicted_batch)) + \
                            ', but type numpy.ndarray is expected.')
        
        if not isinstance(true_batch, np.ndarray):
            raise Exception('Variable `true_batch` has type ' + str(type(true_batch)) + \
                            ', but type numpy.ndarray is expected.')
        
        # check batch shapes
        assert len(predicted_batch.shape) in [2, 3] and len(true_batch.shape) in [2, 3], \
        'Shapes lengths are not correct: ' + str(predicted_batch.shape) + ' and ' + str(true_batch.shape) + ', 2 or 3 are expected.'
        
        # transform shapes
        if len(predicted_batch.shape) == 2:
            predicted_batch = np.reshape(predicted_batch, (predicted_batch.shape[0], predicted_batch.shape[1], 1))
            
        # transform shapes
        if len(true_batch.shape) == 2:
            true_batch = np.reshape(true_batch, (true_batch.shape[0], true_batch.shape[1], 1))
        
        # list for ious
        iou_array = []

        
        # loop over the masks batch
        for i in np.arange(true_batch.shape[2]):
            intersection = np.sum(true_batch[:, :, i].ravel() * predicted_batch[:, :, i].ravel())
            union = np.sum(true_batch[:, :, i].ravel() + predicted_batch[:, :, i].ravel() 
                       - true_batch[:, :, i].ravel() * predicted_batch[:, :, i].ravel())
            iou = (intersection + EPS) / (union + EPS)
            
            iou_array.append(iou)
            
        iou_array = np.array(iou_array)
        
        # return only simgle value if array size is equal to 1
        if iou_array.shape[0] == 1:
            return iou_array[0]
    
        return np.mean(iou_array)
