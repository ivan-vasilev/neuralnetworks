package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import org.junit.Ignore;
import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationAutoencoder;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Environment;

public class AETest {
    
    /**
     * Autoencoder backpropagation
     */
    @Test
    public void testAEBackpropagation() {
	Autoencoder ae = NNFactory.autoencoderSigmoid(6, 2, true);

	TrainingInputProvider trainInputProvider = new SimpleInputProvider(new float[][] {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 1} }, null, 600, 1);
	TrainingInputProvider testInputProvider =  new SimpleInputProvider(new float[][] {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 1} }, new float[][] {{1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1} }, 6, 1);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();
	
	BackPropagationAutoencoder t = TrainerFactory.backPropagationAutoencoder(ae, trainInputProvider, testInputProvider, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.1f, 0.5f, 0f, 0f, 0f);

	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), true, false));

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	t.train();
	ae.removeLayer(ae.getOutputLayer());
	t.test();

	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0);
    }
    
    /**
     * Autoencoder backpropagation
     */
    @Ignore
    @Test
    public void testAEBackpropagation2() {
	Autoencoder ae = NNFactory.autoencoderSigmoid(6, 3, true);
	
	TrainingInputProvider trainInputProvider = new SimpleInputProvider(new float[][] {{1, 1, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 1}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1} }, null, 180000, 1);
	TrainingInputProvider testInputProvider  = new SimpleInputProvider(new float[][] {{1, 1, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 1}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1} }, new float[][] {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 1} }, 9, 1);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();
	
	BackPropagationAutoencoder t = TrainerFactory.backPropagationAutoencoder(ae, trainInputProvider, testInputProvider, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.05f, 0.5f, 0f, 0f, 0f);
	
	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), true, false));
	
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);
	
	t.train();
	ae.removeLayer(ae.getOutputLayer());
	t.test();
	
	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0);
    }
}
