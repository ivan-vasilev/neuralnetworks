package com.github.neuralnetworks.training.events;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardWatchEventKinds;
import java.nio.file.WatchEvent;
import java.nio.file.WatchKey;
import java.nio.file.WatchService;
import java.util.List;

import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;

/**
 * Suspends the current execution if a file named "suspend.txt" is created
 */
public class FileCommandsListener implements TrainingEventListener
{
	private static final long serialVersionUID = 1L;

	private static final String SUSPEND_FILE = "suspend.txt";

	private boolean suspend;

	/**
	 * @param suspendFileFolderPath - path to the folder, where the "suspend.txt" file is located
	 */
	public FileCommandsListener(String suspendFileFolderPath)
	{
		new Thread()
		{
			@Override
			public void run()
			{
				WatchService watchService = null;
				try
				{
					Path path = Paths.get(suspendFileFolderPath);
					watchService = FileSystems.getDefault().newWatchService();
					path.register(watchService, StandardWatchEventKinds.ENTRY_CREATE, StandardWatchEventKinds.ENTRY_DELETE);
				} catch (IOException e1)
				{
					e1.printStackTrace();
				}
				while (true)
				{
					WatchKey watchKey = null;
					try
					{
						watchKey = watchService.take();
					} catch (InterruptedException e)
					{
						e.printStackTrace();
					}

					List<WatchEvent<?>> watchEvents = watchKey.pollEvents();
					for (WatchEvent<?> we : watchEvents)
					{
						if (we.kind() == StandardWatchEventKinds.ENTRY_CREATE && we.context() != null && SUSPEND_FILE.equals(we.context().toString()))
						{
							synchronized (FileCommandsListener.this)
							{
								suspend = true;
							}
						} else if (we.kind() == StandardWatchEventKinds.ENTRY_DELETE && we.context() != null && SUSPEND_FILE.equals(we.context().toString()))
						{
							synchronized (FileCommandsListener.this)
							{
								suspend = false;
								FileCommandsListener.this.notifyAll();
							}
						}
					}
					if (!watchKey.reset())
					{
						break;
					}
				}
			}
		}.start();
	}

	public FileCommandsListener()
	{
		this("");
	}

	@Override
	public void handleEvent(TrainingEvent event)
	{
		if (event instanceof MiniBatchFinishedEvent)
		{
			synchronized (this)
			{
				while (suspend)
				{
					try
					{
						wait();
					} catch (InterruptedException e)
					{
						e.printStackTrace();
					}
				}
			}
		}
	}
}
