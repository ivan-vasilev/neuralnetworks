package com.github.neuralnetworks.training.rbm;

import java.util.List;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.training.DNNLayerTrainer;
import com.github.neuralnetworks.util.Properties;

/**
 * Default implementation for training of Deep Belief Networks
 */
public class DBNTrainer extends DNNLayerTrainer implements TrainingEventListener {

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
		if (current.getMainConnections().getConnectionGraph().getElements().length == next.getMainConnections().getConnectionGraph().getElements().length) {
		    System.arraycopy(current.getMainConnections().getConnectionGraph().getElements(), 0, next.getMainConnections().getConnectionGraph().getElements(), 0, next.getMainConnections().getConnectionGraph().getElements().length);
		}

		if (current.getVisibleBiasConnections() != null && next.getVisibleBiasConnections() != null && current.getVisibleBiasConnections().getConnectionGraph().getElements().length == next.getVisibleBiasConnections().getConnectionGraph().getElements().length) {
		    System.arraycopy(current.getVisibleBiasConnections().getConnectionGraph().getElements(), 0, next.getVisibleBiasConnections().getConnectionGraph().getElements(), 0, next.getVisibleBiasConnections().getConnectionGraph().getElements().length);
		}
		
		if (current.getHiddenBiasConnections() != null && next.getHiddenBiasConnections() != null && current.getHiddenBiasConnections().getConnectionGraph().getElements().length == next.getHiddenBiasConnections().getConnectionGraph().getElements().length) {
		    System.arraycopy(current.getHiddenBiasConnections().getConnectionGraph().getElements(), 0, next.getHiddenBiasConnections().getConnectionGraph().getElements(), 0, next.getHiddenBiasConnections().getConnectionGraph().getElements().length);
		}
	    }
	}
    }
}
