package com.github.neuralnetworks.training.rbm;

import java.util.List;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.training.DNNLayerTrainer;
import com.github.neuralnetworks.util.Properties;

/**
 * Default implementation for training of Deep Belief Networks
 */
public class DBNTrainer extends DNNLayerTrainer implements TrainingEventListener {

    private static final long serialVersionUID = 1L;

    public DBNTrainer(Properties properties) {
	super(properties);
    }

    @Override
    public void handleEvent(TrainingEvent event) {
	// transfer of learned weights from lower to the higher RBM
	if (event instanceof LayerTrainingFinished) {
	    LayerTrainingFinished e = (LayerTrainingFinished) event;
	    CDTrainerBase t = (CDTrainerBase) e.currentTrainer;
	    RBM current = t.getNeuralNetwork();
	    List<? extends NeuralNetwork> list = getNeuralNetwork().getNeuralNetworks();

	    if (list.indexOf(current) < list.size() - 1) {
		RBM next = (RBM) list.get(list.indexOf(current) + 1);
		if (current.getMainConnections().getWeights().getSize() == next.getMainConnections().getWeights().getSize()) {
		    TensorFactory.copy(current.getMainConnections().getWeights(), next.getMainConnections().getWeights());
		}

		if (current.getVisibleBiasConnections() != null && next.getVisibleBiasConnections() != null && current.getVisibleBiasConnections().getWeights().getSize() == next.getVisibleBiasConnections().getWeights().getSize()) {
		    TensorFactory.copy(current.getVisibleBiasConnections().getWeights(), next.getVisibleBiasConnections().getWeights());
		}
		
		if (current.getHiddenBiasConnections() != null && next.getHiddenBiasConnections() != null && current.getHiddenBiasConnections().getWeights().getSize() == next.getHiddenBiasConnections().getWeights().getSize()) {
		    TensorFactory.copy(current.getHiddenBiasConnections().getWeights(), next.getHiddenBiasConnections().getWeights());
		}
	    }
	}
    }
}
