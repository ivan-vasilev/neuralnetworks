package com.github.neuralnetworks.training.rbm;

import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.util.Properties;

/**
 * Aparapi Contrastive Divergence
 */
public class CDAparapiTrainer extends CDAparapiTrainerBase implements TrainingEventListener {

    public CDAparapiTrainer(Properties properties) {
	super(properties);
	addEventListener(this);
    }

    @Override
    public void handleEvent(TrainingEvent event) {
	if (event instanceof SamplingStepEvent) {
	    // clamp results to the visible layer
	    SamplingStepEvent sse = (SamplingStepEvent) event;
	    if (sse.getSamplingCount() == 0) {
		System.arraycopy(getPosPhaseHidden().getElements(), 0, getNegPhaseHidden().getElements(), 0, getNegPhaseHidden().getElements().length);
	    }
	}
    }
}
