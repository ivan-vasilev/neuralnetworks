package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import org.junit.Ignore;
import org.junit.Test;

import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationAutoencoder;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.KernelExecutionStrategy.SeqKernelExecution;

public class AETest {
    
    /**
     * Autoencoder backpropagation
     */
    @Ignore
    
    @Test
    public void testAEBackpropagation() {
	Autoencoder ae = NNFactory.autoencoder(6, 2, true);
	ae.setLayerCalculator(NNFactory.nnSigmoid(ae, null));

	TrainingInputProvider trainInputProvider = new SimpleInputProvider(new float[][] {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 1} }, null, 600, 1);
	TrainingInputProvider testInputProvider =  new SimpleInputProvider(new float[][] {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 1} }, new float[][] {{1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1} }, 6, 1);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();
	
	BackPropagationAutoencoder t = TrainerFactory.backPropagationSigmoidAutoencoder(ae, trainInputProvider, testInputProvider, error, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.1f, 0.5f, 0f, 0f);

	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), true, false));

	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());

	t.train();
	ae.removeLayer(ae.getOutputLayer());
	t.test();

	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0);
    }
    
    /**
     * Autoencoder backpropagation
     */
    @Test
    public void testAEBackpropagation2() {
	Autoencoder ae = NNFactory.autoencoder(6, 3, true);
	ae.setLayerCalculator(NNFactory.nnSigmoid(ae, null));
	
	TrainingInputProvider trainInputProvider = new SimpleInputProvider(new float[][] {{1, 1, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 1}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1} }, null, 180000, 1);
	TrainingInputProvider testInputProvider  = new SimpleInputProvider(new float[][] {{1, 1, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 1}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1} }, new float[][] {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 1} }, 9, 1);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();
	
	BackPropagationAutoencoder t = TrainerFactory.backPropagationSigmoidAutoencoder(ae, trainInputProvider, testInputProvider, error, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.05f, 0.5f, 0f, 0f);
	
	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), true, false));
	
	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());
	
	t.train();
	ae.removeLayer(ae.getOutputLayer());
	t.test();
	
	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0);
    }
}
