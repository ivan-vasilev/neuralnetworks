package com.github.neuralnetworks.ui;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.BreadthFirstOrderStrategy;
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.Tensor.TensorIterator;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.MiniBatchFinishedEvent;
import com.github.neuralnetworks.training.events.TestingFinishedEvent;
import com.github.neuralnetworks.training.events.TestingStartedEvent;
import com.github.neuralnetworks.training.events.TrainingFinishedEvent;
import com.github.neuralnetworks.training.events.TrainingStartedEvent;
import com.github.neuralnetworks.util.Util;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.ActionEvent;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.ScrollPane.ScrollBarPolicy;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

/**
 * ui listener for feedfowrad neural networks with backpropagation
 */
@SuppressWarnings({ "unchecked", "rawtypes" })
public class FFNBPCalculationEventListener implements TrainingEventListener
{

	private static final long serialVersionUID = 1L;

	private static final int errorDisplayPoints = 100;

	/**
	 * refresh time (milisec)
	 */
	private long refreshInterval;

	public FFNBPCalculationEventListener()
	{
		this(500);
	}

	public FFNBPCalculationEventListener(int refreshInterval)
	{
		super();

		this.refreshInterval = refreshInterval;
	}

	@Override
	public void handleEvent(TrainingEvent event)
	{
		NNCalculationData n = NNCalculationData.getInstance();
		if (n.trainer == null)
		{
			n.trainer = (Trainer) event.getSource();
		}

		if (event instanceof TrainingStartedEvent)
		{
			n.isTraining = true;
			reset();
			n.lastRefreshTime = n.startTime = System.currentTimeMillis();
		} else if (event instanceof TestingStartedEvent)
		{
			n.isTraining = false;
			reset();
			n.lastRefreshTime = n.startTime = System.currentTimeMillis();
		} else if (event instanceof TrainingFinishedEvent)
		{
			n.isTraining = false;
			Platform.runLater(() -> n.controlButtonCaption.set("Start testing"));

			n.trainingFinishedLatch = new CountDownLatch(1);
			try
			{
				n.trainingFinishedLatch.await();
			} catch (InterruptedException ex)
			{
				throw new RuntimeException("Unexpected exception: ", ex);
			}
		} else if (event instanceof TestingFinishedEvent)
		{
			Platform.runLater(() -> n.controlButtonCaption.set("Close"));

			n.testingFinishedLatch = new CountDownLatch(1);
			try
			{
				n.testingFinishedLatch.await();
			} catch (InterruptedException ex)
			{
				throw new RuntimeException("Unexpected exception: ", ex);
			}
		}

		if (event instanceof MiniBatchFinishedEvent)
		{
			launch();

			MiniBatchFinishedEvent mbe = (MiniBatchFinishedEvent) event;
			updateError(mbe);
			updateNetwork(mbe);
			updateFooter(mbe);
			waitSample();

			if (refresh(mbe))
			{
				n.lastRefreshTime = System.currentTimeMillis();
			}
		}
	}

	private boolean refresh(MiniBatchFinishedEvent mbe)
	{
		NNCalculationData n = NNCalculationData.getInstance();

		BackPropagationTrainer t = (BackPropagationTrainer) mbe.getSource();
		boolean last = false;
		if (isTraining())
		{
			last = mbe.getBatchCount() * t.getTrainingBatchSize() - t.getTrainingInputProvider().getInputSize() * t.getEpochs() == 0 ? true : false;
		} else if (isTesting())
		{
			last = mbe.getBatchCount() * t.getTestBatchSize() - t.getTestingInputProvider().getInputSize() == 0 ? true : false;
		}

		return System.currentTimeMillis() - n.lastRefreshTime >= refreshInterval || n.pauseOnEachSample.get() || last;
	}

	private void updateError(MiniBatchFinishedEvent mbe)
	{
		BackPropagationTrainer t = (BackPropagationTrainer) mbe.getSource();

		if (isTraining())
		{
			// max 100 data points
			int size = 0;
			float batchSize = 0;
			if (isTraining())
			{
				size = t.getTrainingInputProvider().getInputSize() * t.getEpochs();
				batchSize = t.getTrainingBatchSize();
			} else if (isTesting())
			{
				size = t.getTestingInputProvider().getInputSize();
				batchSize = t.getTestBatchSize();
			}

			int stepSize = 0;
			if (size / batchSize > errorDisplayPoints)
			{
				stepSize = size / errorDisplayPoints;
			} else
			{
				stepSize = (int) (size / batchSize);
			}

			int currentBatch = mbe.getBatchCount();

			NNCalculationData d = NNCalculationData.getInstance();
			Tensor activations = mbe.getResults().get(t.getNeuralNetwork().getOutputLayer());
			Tensor target = mbe.getResults().get(t.getLossFunction());

			float error = t.getLossFunction().getLossFunction(activations, target);

			if (currentBatch * batchSize - d.series.size() * stepSize >= stepSize)
			{
				d.series.add(new float[] { currentBatch, error });
			}

			if (refresh(mbe) && d.chartSeries.getData().size() < d.series.size())
			{
				Platform.runLater(() -> {
					NNCalculationData.getInstance().currentError.set((isTraining() ? "Testing " : "Training ") + "error: " + error);
					d.chartSeries.getData().clear();
					synchronized (d.series)
					{
						for (float[] s : d.series)
						{
							d.chartSeries.getData().add(new XYChart.Data(s[0], s[1]));
						}
					}
				});
			}
		} else if (isTesting() && refresh(mbe))
		{
			Platform.runLater(() -> NNCalculationData.getInstance().currentError.set("error: " + t.getOutputError().getTotalNetworkError()));
		}
	}

	private void updateNetwork(MiniBatchFinishedEvent mbe)
	{
		if (refresh(mbe))
		{
			CountDownLatch latch = new CountDownLatch(1);
			Platform.runLater(() -> {
				NNCalculationData nd = NNCalculationData.getInstance();
				for (NNCalculationData.NetworkRowData n : nd.networkRows)
				{
					// update min/max values
					float min = Float.MAX_VALUE, max = Float.MIN_VALUE, avg = 0;
					TensorIterator it = n.activations.iterator();
					while (it.hasNext())
					{
						int i = it.next();
						if (min > n.activations.getElements()[i])
						{
							min = n.activations.getElements()[i];
						}

						if (max < n.activations.getElements()[i])
						{
							max = n.activations.getElements()[i];
						}

						avg += n.activations.getElements()[i];
					}
					avg /= n.activations.getSize();

					n.minMaxAvgActivations.set("MIN/AVG/MAX   " + String.format("%.3f", min) + "/" + String.format("%.3f", avg) + "/" + String.format("%.3f", max));
					if (n.connection instanceof FullyConnected)
					{
						TensorIterator i = n.activations.iterator();
						for (int j = 0; i.hasNext(); j++)
						{
							double pixel = 1 - (n.activations.getElements()[i.next()] - min) / (max - min);
							pixel = pixel > 1 ? 1 : pixel;
							pixel = pixel < 0 ? 0 : pixel;
							for (int k = 0; k < n.pixelSize; k++)
							{
								for (int l = 0; l < n.pixelSize; l++)
								{
									n.activationPixelWriters.get(j).setColor(k, l, Color.color(pixel, pixel, pixel, 1));
								}
							}
						}
					} else if (n.connection instanceof Conv2DConnection || n.connection instanceof Subsampling2DConnection)
					{
						for (int j = 0; j < n.activationPixelWriters.size(); j++)
						{
							TensorIterator i = n.activations.iterator(new int[][] { { 0, j, 0, 0 }, { 0, j, n.activations.getDimensions()[2] - 1, n.activations.getDimensions()[3] - 1 } });
							for (int k = 0; k < n.activations.getDimensions()[1]; k++)
							{
								for (int l = 0; l < n.activations.getDimensions()[3]; l++)
								{
									double pixel = 1 - (n.activations.getElements()[i.next()] - min) / (max - min);
									pixel = pixel > 1 ? 1 : pixel;
									pixel = pixel < 0 ? 0 : pixel;

									for (int q = 0; q < n.pixelSize; q++)
									{
										for (int r = 0; r < n.pixelSize; r++)
										{
											n.activationPixelWriters.get(j).setColor(l * n.pixelSize + q, k * n.pixelSize + r, Color.color(pixel, pixel, pixel, 1));
										}
									}
								}
							}
						}

						// weights
						if (n.connection instanceof Conv2DConnection && n.weightsPixelWriters.size() > 0)
						{
							Conv2DConnection con = (Conv2DConnection) n.connection;

							min = Float.MAX_VALUE;
							max = Float.MIN_VALUE;
							it = con.getWeights().iterator();
							while (it.hasNext())
							{
								int i = it.next();
								if (min > con.getWeights().getElements()[i])
								{
									min = con.getWeights().getElements()[i];
								}

								if (max < con.getWeights().getElements()[i])
								{
									max = con.getWeights().getElements()[i];
								}
							}

							for (int j = 0; j < con.getOutputFilters(); j++)
							{
								for (int p = 0; p < con.getInputFilters(); p++)
								{
									TensorIterator i = con.getWeights().iterator(new int[][] { { j, p, 0, 0 }, { j, p, con.getFilterRows() - 1, con.getFilterColumns() - 1 } });
									for (int k = 0; k < con.getWeights().getDimensions()[2]; k++)
									{
										for (int l = 0; l < con.getWeights().getDimensions()[3]; l++)
										{
											double pixel = 1 - (con.getWeights().getElements()[i.next()] - min) / (max - min);
											pixel = pixel > 1 ? 1 : pixel;
											pixel = pixel < 0 ? 0 : pixel;

											for (int q = 0; q < n.pixelSize; q++)
											{
												for (int r = 0; r < n.pixelSize; r++)
												{
													n.weightsPixelWriters.get(j).setColor(l * n.pixelSize + q, k * n.pixelSize + r, Color.color(pixel, pixel, pixel, 1));
												}
											}
										}
									}
								}
							}
						}
					}
				}

				// target
				BackPropagationTrainer<NeuralNetwork> bpt = (BackPropagationTrainer<NeuralNetwork>) nd.trainer;
				Tensor target = mbe.getResults().get(bpt.getLossFunction());

				// update min/max values
				float min = Float.MAX_VALUE;
				float max = Float.MIN_VALUE;

				TensorIterator it = target.iterator();
				while (it.hasNext())
				{
					int i = it.next();
					if (min > target.getElements()[i])
					{
						min = target.getElements()[i];
					}

					if (max < target.getElements()[i])
					{
						max = target.getElements()[i];
					}
				}

				float minF = min, maxF = max;
				TensorIterator i = target.iterator();
				int pixelSize = nd.networkRows.get(nd.networkRows.size() - 1).pixelSize;
				for (int j = 0; i.hasNext(); j++)
				{
					double pixel = 1 - (target.getElements()[i.next()] - minF) / (maxF - minF);
					pixel = pixel > 1 ? 1 : pixel;
					pixel = pixel < 0 ? 0 : pixel;
					for (int k = 0; k < pixelSize; k++)
					{
						for (int l = 0; l < pixelSize; l++)
						{
							nd.targetPixelWriters.get(j).setColor(k, l, Color.color(pixel, pixel, pixel, 1));
						}
					}
				}

				latch.countDown();
			});

			try
			{
				latch.await();
			} catch (InterruptedException ex)
			{
				throw new RuntimeException("Unexpected exception: ", ex);
			}
		}
	}

	private void updateFooter(MiniBatchFinishedEvent event)
	{
		if (refresh(event))
		{
			Platform.runLater(() -> {
				NNCalculationData n = NNCalculationData.getInstance();

				Trainer t = (Trainer) event.getSource();

				float time = System.currentTimeMillis() - n.startTime;

				int totalSize = 0;
				if (isTraining())
				{
					totalSize = (t.getTrainingInputProvider().getInputSize() * t.getEpochs()) / t.getTrainingBatchSize();
				} else if (isTesting())
				{
					totalSize = t.getTestingInputProvider().getInputSize() / t.getTestBatchSize();
				}

				n.time.set("Time: " + String.format("%.1f", time / 1000) + "s\nMB: " + event.getBatchCount() + "/" + totalSize + ", " + String.format("%.1f", time / event.getBatchCount()) + "ms per MB");

				Hyperparameters hp = t.getHyperparameters();
				NNCalculationData.getInstance().trainingParams.set("LR: " + hp.getDefaultLearningRate() + "; M: " + hp.getDefaultMomentum() + "; L1: "
						+ hp.getDefaultL1WeightDecay() + "; L2: " + hp.getDefaultL2WeightDecay());
			});
		}
	}

	private boolean isTraining()
	{
		return NNCalculationData.getInstance().isTraining;
	}

	private boolean isTesting()
	{
		return !NNCalculationData.getInstance().isTraining;
	}

	/**
	 * launch the application
	 */
	private void launch()
	{
		if (NNCalculationData.getInstance().launchLatch.getCount() > 0)
		{
			new Thread()
			{
				@Override
				public synchronized void run()
				{
					javafx.application.Application.launch(NNCalculationApplication.class);
				}
			}.start();
		}

		try
		{
			NNCalculationData.getInstance().launchLatch.await();
		} catch (InterruptedException ex)
		{
			throw new RuntimeException("Unexpected exception: ", ex);
		}
	}

	/**
	 * wait to press button for each sample
	 */
	private void waitSample()
	{
		NNCalculationData n = NNCalculationData.getInstance();
		if (n.pauseOnEachSample.get())
		{
			try
			{
				n.pauseOnEachSampleLatch.await();
			} catch (InterruptedException ex)
			{
				throw new RuntimeException("Unexpected exception: ", ex);
			}

			n.pauseOnEachSampleLatch = new CountDownLatch(1);
		} else
		{
			try
			{
				n.pauseLatch.await();
			} catch (InterruptedException ex)
			{
				throw new RuntimeException("Unexpected exception: ", ex);
			}
		}
	}

	protected void reset()
	{
		NNCalculationData n = NNCalculationData.getInstance();
		n.startTime = n.lastRefreshTime = 0;
		NNCalculationData.getInstance().series.clear();
	}

	public static class NNCalculationApplication extends Application
	{

		private BooleanProperty displayLossFunction = new SimpleBooleanProperty(true);
		private BooleanProperty displayNetwork = new SimpleBooleanProperty(true);

		@Override
		public void start(Stage primaryStage) throws Exception
		{
			primaryStage.setTitle("Neural network");

			BorderPane border = new BorderPane();

			border.setTop(addHeader());
			border.setCenter(addCenter());
			border.setBottom(addFooter());

			Scene scene = new Scene(border, 1024, 768);
			primaryStage.setScene(scene);
			primaryStage.show();

			NNCalculationData.getInstance().launchLatch.countDown();
		}

		private ScrollPane addCenter()
		{
			VBox vbox = new VBox();
			vbox.getChildren().addAll(addLossFunction(), addNetwork());

			ScrollPane sp = new ScrollPane();
			sp.setHbarPolicy(ScrollBarPolicy.NEVER);
			sp.setVbarPolicy(ScrollBarPolicy.AS_NEEDED);
			sp.setFitToWidth(true);
			sp.setFitToHeight(true);
			sp.setContent(vbox);

			return sp;
		}

		private Pane addHeader()
		{
			HBox hbox = new HBox();
			hbox.setPadding(new Insets(15, 12, 15, 12));
			hbox.setSpacing(10);
			hbox.setStyle("-fx-background-color: #336699;");

			HBox left = new HBox();
			left.setAlignment(Pos.CENTER_LEFT);
			left.setSpacing(10);

			NNCalculationData n = NNCalculationData.getInstance();
			Button startStop = new Button();
			startStop.textProperty().bind(n.controlButtonCaption);
			n.controlButtonCaption.set("Stop");

			startStop.setOnAction((ActionEvent e) -> {
				if (n.trainingFinishedLatch.getCount() > 0)
				{
					n.trainingFinishedLatch.countDown();
					n.controlButtonCaption.set(n.pauseOnEachSample.get() ? "Continue" : "Stop");
				} else if (n.testingFinishedLatch.getCount() > 0)
				{
					n.testingFinishedLatch.countDown();
					n.controlButtonCaption.set(n.pauseOnEachSample.get() ? "Continue" : "Stop");
				} else if (n.pauseOnEachSample.get())
				{
					n.pauseOnEachSampleLatch.countDown();
				} else if (n.pauseLatch.getCount() > 0)
				{
					startStop.setText("Stop");
					n.pauseLatch.countDown();
				} else
				{
					n.controlButtonCaption.set("Continue");
					n.pauseLatch.countDown();
					n.pauseLatch = new CountDownLatch(1);
				}
			});
			startStop.setPrefSize(100, 20);

			CheckBox pauseOnEachSample = new CheckBox();
			pauseOnEachSample.setStyle("-fx-text-fill: white;");
			pauseOnEachSample.selectedProperty().bindBidirectional(n.pauseOnEachSample);
			pauseOnEachSample.selectedProperty().addListener(new ChangeListener<Boolean>()
			{
				@Override
				public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue)
				{
					n.pauseLatch.countDown();
					n.pauseOnEachSampleLatch.countDown();
					if (newValue)
					{
						n.pauseOnEachSampleLatch = new CountDownLatch(1);
						n.controlButtonCaption.set("Continue");
					} else
					{
						n.pauseOnEachSampleLatch = new CountDownLatch(0);
						n.controlButtonCaption.set("Stop");
					}
				}
			});
			pauseOnEachSample.setText("Pause on each sample");

			CheckBox showHideError = new CheckBox();
			showHideError.selectedProperty().bindBidirectional(displayLossFunction);
			showHideError.setText("Display loss function");
			showHideError.setStyle("-fx-text-fill: white;");

			CheckBox showHideNetwork = new CheckBox();
			showHideNetwork.selectedProperty().bindBidirectional(displayNetwork);
			showHideNetwork.setText("Display network");
			showHideNetwork.setStyle("-fx-text-fill: white;");

			left.getChildren().addAll(startStop, pauseOnEachSample, showHideError, showHideNetwork);

			HBox right = new HBox();
			right.setAlignment(Pos.CENTER_RIGHT);
			right.setSpacing(3);
			HBox.setHgrow(right, Priority.ALWAYS);

			Label error = new ControlLabel();
			error.textProperty().bind(NNCalculationData.getInstance().currentError);

			right.getChildren().addAll(error);

			hbox.getChildren().addAll(left, right);

			return hbox;
		}

		private Pane addFooter()
		{
			HBox hbox = new HBox();
			hbox.setPadding(new Insets(5, 12, 5, 12));
			hbox.setSpacing(10);
			hbox.setStyle("-fx-background-color: #336699;");

			Label time = new ControlLabel();
			time.textProperty().bind(NNCalculationData.getInstance().time);

			HBox right = new HBox();
			right.setAlignment(Pos.CENTER_RIGHT);
			right.setSpacing(3);
			HBox.setHgrow(right, Priority.ALWAYS);

			Label trainingParams = new ControlLabel();
			trainingParams.textProperty().bind(NNCalculationData.getInstance().trainingParams);

			right.getChildren().addAll(trainingParams);

			hbox.getChildren().addAll(time, right);

			return hbox;
		}

		private LineChart<Number, Number> addLossFunction()
		{
			NumberAxis xAxis = new NumberAxis();
			xAxis.setLabel("Mini batch");
			NumberAxis yAxis = new NumberAxis();

			LineChart<Number, Number> lineChart = new LineChart<Number, Number>(xAxis, yAxis);

			lineChart.setTitle("Error");
			lineChart.setLegendVisible(false);
			lineChart.visibleProperty().bindBidirectional(displayLossFunction);
			lineChart.managedProperty().bind(lineChart.visibleProperty());

			NNCalculationData.getInstance().chartSeries = new XYChart.Series();

			lineChart.getData().add(NNCalculationData.getInstance().chartSeries);

			return lineChart;
		}

		private Pane addNetwork()
		{
			NNCalculationData n = NNCalculationData.getInstance();
			NeuralNetwork nn = n.trainer.getNeuralNetwork();

			Set<Layer> calculatedLayers = new HashSet<>();
			calculatedLayers.add(nn.getInputLayer());
			List<ConnectionCandidate> ccc = new BreadthFirstOrderStrategy(n.trainer.getNeuralNetwork(), n.trainer.getNeuralNetwork().getOutputLayer()).order();
			Collections.reverse(ccc);

			ccc.removeIf(i -> Util.isBias(Util.getOppositeLayer(i.connection, i.target)));

			ccc.add(0, new ConnectionCandidate(ccc.get(0).connection, ccc.get(0).connection.getInputLayer()));

			VBox vbox = new VBox();
			vbox.visibleProperty().bindBidirectional(displayNetwork);

			for (ConnectionCandidate c : ccc)
			{
				Pane row = addNetworkRow(c);
				if (ccc.indexOf(c) % 2 == 0)
				{
					row.setStyle("-fx-background-color: #FFFF99; -fx-padding: 5; -fx-font-size: 15");
				} else
				{
					row.setStyle("-fx-padding: 5; -fx-font-size: 15");
				}

				vbox.getChildren().add(row);
			}

			return vbox;
		}

		private Pane addNetworkRow(ConnectionCandidate c)
		{
			NNCalculationData n = NNCalculationData.getInstance();
			BackPropagationTrainer t = (BackPropagationTrainer) n.trainer;
			NeuralNetwork nn = t.getNeuralNetwork();

			// layer info
			String layerInfo = "";

			// input
			boolean isInput = c.target == nn.getInputLayer();
			if (isInput)
			{
				layerInfo += "INPUT ";
			} else if (c.target == nn.getOutputLayer())
			{
				layerInfo += "OUTPUT ";
			}

			if (c.connection instanceof FullyConnected)
			{
				layerInfo += "FC " + (c.target == c.connection.getOutputLayer() ? c.connection.getOutputUnitCount() : c.connection.getInputUnitCount());
				if (!isInput)
				{
					FullyConnected fc = (FullyConnected) c.connection;
					layerInfo += ", weights " + fc.getWeights().getColumns() + "x" + fc.getWeights().getRows();
					if (Util.hasBias(c.target.getConnections()))
					{
						layerInfo += ", params " + (fc.getWeights().getColumns() + 1) * fc.getWeights().getRows();
					} else
					{
						layerInfo += ", params " + fc.getWeights().getColumns() * fc.getWeights().getRows();
					}
				}
			} else if (c.connection instanceof Conv2DConnection)
			{
				Conv2DConnection cn = (Conv2DConnection) c.connection;
				if (c.target == c.connection.getInputLayer())
				{
					layerInfo +=
							"CONV " + cn.getInputFeatureMapRows() + "x" + cn.getInputFeatureMapColumns() + "x" + cn.getInputFilters();
				} else
				{
					layerInfo +=
							"CONV " + cn.getOutputFeatureMapRows() + "x" + cn.getOutputFeatureMapColumns() + "x" + cn.getOutputFilters() +
									", filter " + cn.getFilterRows() + "x" + cn.getFilterColumns() + "x" + cn.getOutputFilters();
					int params = cn.getInputFilters() * cn.getOutputFilters() * cn.getFilterRows() * cn.getFilterColumns();
					if (Util.hasBias(c.target.getConnections()))
					{
						params += cn.getOutputFilters();
					}
					layerInfo += ", params " + params + ", rowStride " + cn.getRowStride() + ", columnStride " + cn.getColumnStride();
				}
			} else if (c.connection instanceof Subsampling2DConnection)
			{
				Subsampling2DConnection cn = (Subsampling2DConnection) c.connection;
				if (c.target == c.connection.getInputLayer())
				{
					layerInfo += "POOL " + cn.getInputFeatureMapRows() + "x" + cn.getInputFeatureMapColumns() + "x" + cn.getFilters();
				} else
				{
					layerInfo += "POOL " + cn.getOutputFeatureMapRows() + "x" + cn.getOutputFeatureMapColumns() + "x" + cn.getFilters();
					layerInfo += ", pooling region " + cn.getSubsamplingRegionRows() + "x" + cn.getSubsamplingRegionCols();
				}
			}

			// min/max activations
			NNCalculationData.NetworkRowData nd = new NNCalculationData.NetworkRowData(c.connection, TensorFactory.tensor(c.target, c.connection, t.getActivations()));
			n.networkRows.add(nd);

			Label minMaxActivations = new Label();
			minMaxActivations.textProperty().bind(nd.minMaxAvgActivations);

			VBox left = new VBox();
			left.setPrefWidth(400);
			left.getChildren().addAll(new Label(layerInfo), minMaxActivations);

			// activation/weight images
			VBox right = new VBox();
			right.setAlignment(Pos.CENTER_LEFT);
			HBox.setHgrow(right, Priority.ALWAYS);
			right.getChildren().add(new Label("Activations"));

			FlowPane activationImages = new FlowPane();
			activationImages.setHgap(3);
			activationImages.setVgap(3);
			right.getChildren().add(activationImages);

			if (c.connection instanceof FullyConnected)
			{
				nd.pixelSize = 8;
				for (int i = 0; i < nd.activations.getSize(); i++)
				{
					WritableImage image = new WritableImage(nd.pixelSize, nd.pixelSize);
					nd.activationPixelWriters.add(image.getPixelWriter());
					activationImages.getChildren().add(new ImageView(image));
				}
			} else if (c.connection instanceof Conv2DConnection || c.connection instanceof Subsampling2DConnection)
			{
				nd.pixelSize = 2;
				int width = 0, height = 0, images = 0;
				if (c.target == c.connection.getInputLayer() && c.connection instanceof Conv2DConnection)
				{
					Conv2DConnection con = (Conv2DConnection) c.connection;
					width = con.getInputFeatureMapRows();
					height = con.getInputFeatureMapColumns();
					images = con.getInputFilters();
				} else if (c.target == c.connection.getInputLayer() && c.connection instanceof Subsampling2DConnection)
				{
					Subsampling2DConnection con = (Subsampling2DConnection) c.connection;
					width = con.getInputFeatureMapRows();
					height = con.getInputFeatureMapColumns();
					images = con.getFilters();
				} else if (c.target == c.connection.getOutputLayer() && c.connection instanceof Conv2DConnection)
				{
					Conv2DConnection con = (Conv2DConnection) c.connection;
					width = con.getOutputFeatureMapRows();
					height = con.getOutputFeatureMapColumns();
					images = con.getOutputFilters();
				} else if (c.target == c.connection.getOutputLayer() && c.connection instanceof Subsampling2DConnection)
				{
					Subsampling2DConnection con = (Subsampling2DConnection) c.connection;
					width = con.getOutputFeatureMapRows();
					height = con.getOutputFeatureMapColumns();
					images = con.getFilters();
				}

				for (int i = 0; i < images; i++)
				{
					WritableImage image = new WritableImage(width * nd.pixelSize, height * nd.pixelSize);
					nd.activationPixelWriters.add(image.getPixelWriter());
					activationImages.getChildren().add(new ImageView(image));
				}

				// add weights
				if (c.connection instanceof Conv2DConnection && c.target == c.connection.getOutputLayer())
				{
					Conv2DConnection con = (Conv2DConnection) c.connection;
					right.getChildren().add(new Label("Weights"));
					FlowPane weightsImages = new FlowPane();
					weightsImages.setHgap(3);
					weightsImages.setVgap(3);
					right.getChildren().add(weightsImages);

					for (int i = 0; i < con.getInputFilters() * con.getOutputFilters(); i++)
					{
						WritableImage image = new WritableImage(con.getFilterRows() * nd.pixelSize, con.getFilterColumns() * nd.pixelSize);
						nd.weightsPixelWriters.add(image.getPixelWriter());
						weightsImages.getChildren().add(new ImageView(image));
					}
				}
			}

			// target layer
			if (c.target == nn.getOutputLayer())
			{
				right.getChildren().add(new Label("Target"));

				FlowPane targetImages = new FlowPane();
				targetImages.setHgap(3);
				targetImages.setVgap(3);
				right.getChildren().add(targetImages);

				for (int i = 0; i < nd.activations.getSize(); i++)
				{
					WritableImage image = new WritableImage(nd.pixelSize, nd.pixelSize);
					n.targetPixelWriters.add(image.getPixelWriter());
					targetImages.getChildren().add(new ImageView(image));
				}
			}

			HBox hbox = new HBox();
			hbox.getChildren().addAll(left, right);

			return hbox;
		}

		private static class ControlLabel extends Label
		{

			public ControlLabel()
			{
				super();
				setStyle("-fx-text-fill: white;");
			}
		}
	}

	private static class NNCalculationData
	{

		private static NNCalculationData instance = new NNCalculationData();
		private XYChart.Series chartSeries;
		private List<float[]> series;
		private CountDownLatch launchLatch;
		private BooleanProperty pauseOnEachSample;
		private CountDownLatch pauseOnEachSampleLatch;
		private CountDownLatch pauseLatch;
		private CountDownLatch trainingFinishedLatch;
		private CountDownLatch testingFinishedLatch;
		private StringProperty currentError;
		private StringProperty time;
		private StringProperty trainingParams;
		private StringProperty controlButtonCaption;
		private long startTime;
		private long lastRefreshTime;
		private boolean isTraining;
		private Trainer trainer;
		private List<NetworkRowData> networkRows;
		private List<PixelWriter> targetPixelWriters;

		private NNCalculationData()
		{
			super();
			launchLatch = new CountDownLatch(1);
			pauseOnEachSampleLatch = new CountDownLatch(0);
			trainingFinishedLatch = new CountDownLatch(1);
			testingFinishedLatch = new CountDownLatch(1);
			pauseLatch = new CountDownLatch(0);
			pauseOnEachSample = new SimpleBooleanProperty(false);
			controlButtonCaption = new SimpleStringProperty();
			currentError = new SimpleStringProperty();
			time = new SimpleStringProperty();
			trainingParams = new SimpleStringProperty();
			series = Collections.synchronizedList(new ArrayList<>());
			networkRows = Collections.synchronizedList(new ArrayList<>());
			targetPixelWriters = new ArrayList<>();
		}

		public static NNCalculationData getInstance()
		{
			return instance;
		}

		public static class NetworkRowData
		{
			private Connections connection;
			private Tensor activations;
			private StringProperty minMaxAvgActivations;
			private List<PixelWriter> activationPixelWriters;
			private List<PixelWriter> weightsPixelWriters;
			private int pixelSize;

			public NetworkRowData(Connections connection, Tensor activations)
			{
				super();
				this.connection = connection;
				this.activations = activations;
				this.minMaxAvgActivations = new SimpleStringProperty();
				this.activationPixelWriters = new ArrayList<>();
				this.weightsPixelWriters = new ArrayList<>();
			}
		}
	}
}
