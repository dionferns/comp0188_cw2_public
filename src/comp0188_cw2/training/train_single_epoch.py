
import torch
from typing import Dict, Tuple
import logging
from tqdm import tqdm
from pymlrf.types import CriterionProtocol
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
from pymlrf.types import (
    GenericDataLoaderProtocol
    )

class TrainSingleEpoch:
    
    def __init__(
        self, 
        half_precision:bool=False,
        cache_preds:bool=True,
        enable_grad_clipping: bool = False  # Default value set to False
        ) -> None:
        """Class which runs a single epoch of training.

        Args:
            half_precision (bool, optional): Boolean defining whether to run in 
            half-precision. Defaults to False.
            cache_preds (bool, optional): Boolean defining whether to save the 
            prediction outputs. Defaults to True.
        """
        self.half_precision = half_precision
        self.cache_preds = cache_preds
        self.enable_grad_clipping = enable_grad_clipping  # Store the new parameter
        
    def __call__(
        self,
        model:torch.nn.Module,
        data_loader:GenericDataLoaderProtocol,
        gpu:bool,
        optimizer:torch.optim.Optimizer,
        criterion:CriterionProtocol,
        logger:logging.Logger
        )->Tuple[torch.Tensor, Dict[str,torch.Tensor]]:
        """ Call function which runs a single epoch of training
        Args:
            model (BaseModel): Torch model of type BaseModel i.e., it should
            subclass the BaseModel class
            data_loader (DataLoader): Torch data loader object
            gpu (bool): Boolean defining whether to use a GPU if available
            optimizer (torch.optim.Optimizer): Torch optimiser to use in training
            criterion (CriterionProtocol): Criterian to use for training or 
            for training and validation if val_criterion is not specified. I.e., 
            this could be nn.MSELoss() 
            logger (logging.Logger): Logger object to use for printing to terminal
        Raises:
            RuntimeError: Captures generic runtime errors that may occur during 
            training

        Returns:
            Tuple[torch.Tensor, Dict[str,torch.Tensor]]: Tuple defining the 
            final loss for the epoch and a dictionary of predictions. The keys 
            will be the same keys required by the criterion. 
        """

	#Change 1
        losses = torch.tensor(0.0)
        denom = torch.tensor(0.0)
        if gpu:
            _device = "cuda"
        else:
            _device = "cpu"
        #Change 2.
        losses = losses.to(_device)
        denom = denom.to(_device)
        if self.half_precision:
            losses = losses.half()
            denom = denom.half()
        model.train()
        
        preds = []
        range_gen = tqdm(
            enumerate(data_loader),
            total=len(data_loader)
            #desc=f"Epoch {int(epoch)}/{epochs}",
            )
        for i, vals in range_gen:

            input_vals = vals.input
            output_vals = vals.output
            if gpu:
                input_vals = {
                    val:input_vals[val].cuda() for val in input_vals
                    }
                output_vals = {
                    val:output_vals[val].cuda() for val in output_vals
                    }
            else:
                input_vals = {val:Variable(input_vals[val]) for val in input_vals}
                output_vals = {val:Variable(output_vals[val])
                            for val in output_vals}
            
            #output_vals["grp"] = output_vals["grp"].argmax(dim=1)
            optimizer.zero_grad()

            # Compute output
            if self.half_precision:
                with torch.autocast(device_type=_device):
                        output = model(**input_vals)
                    

                        #changes made here.
                        train_loss = criterion(output, output_vals)
                        # Separate losses for pos and grp
                        #pos_loss = criterion.loss_lkp["pos"](output["pos"], output_vals["pos"])  # Added
                        #grp_loss = criterion.loss_lkp["grp"](output["grp"], output_vals["grp"])  # Adde
                    
                        # Combine losses using weights
                        #alpha = 1.0    #(1.0 + pos_loss.item()) Weight for pos_loss
                        #beta = 5.0 #(1.0 + grp_loss.item()) Weight for grp_loss
                        #train_loss = alpha * pos_loss + beta * grp_loss  # Weighted loss added here




            else:
                output = model(**input_vals)
                

                #chagnes made here.
                # Separate losses for pos and grp
                #pos_loss = criterion.loss_lkp["pos"](output["pos"], output_vals["pos"])  # Added
                #grp_loss = criterion.loss_lkp["grp"](output["grp"], output_vals["grp"])  # Added

                # Combine losses using weights
                #alpha = 1.0 #(1.0 + pos_loss.item()) Weight for pos_loss
                #beta = 5.0 #(1.0 + grp_loss.item()) Weight for grp_loss
                #train_loss = alpha * pos_loss + beta * grp_loss  # Weighted loss added here
                train_loss = criterion(output, output_vals)
            if self.cache_preds:
                preds.append({k:output[k].detach().cpu() for k in output.keys()})
            

	    #change 3.
            losses += train_loss.detach().to(_device)

            denom += 1
            # losses.update(train_loss.data[0], g.size(0))
            # error_ratio.update(evaluation(output, target).data[0], g.size(0))

            try:
                # compute gradient and do SGD step
                train_loss.backward()
                

                #change made here.
                if self.enable_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

                optimizer.step()
            except RuntimeError as e:
                logger.debug("Runtime error on training instance: {}".format(i))
                raise e
        _prd_lst = {}
        if self.cache_preds:
            for k in preds[0].keys():
                _prd_lst[k] = torch.concat([t[k] for t in preds],dim=0)
        losses = losses/denom
        return losses, _prd_lst
