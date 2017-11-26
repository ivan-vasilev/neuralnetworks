package com.github.neuralnetworks.input;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;

/**
 * Input provider that takes float arrays from file
 */
public class FileBatchReader
{
	private RandomAccessFile file;
	private byte[] buffer;

	public FileBatchReader(RandomAccessFile file, int batchLength)
	{
		this.buffer = new byte[batchLength];
		this.file = file;
	}

	public byte[] getNextInput()
	{
		try
		{
			if (file.length() - file.getFilePointer() < buffer.length)
			{
				int s = (int) (file.length() - file.getFilePointer());
				ByteBuffer bb = ByteBuffer.wrap(buffer, 0, s);
				this.file.getChannel().read(bb);

				bb = ByteBuffer.wrap(buffer, buffer.length - s, s);
				this.file.seek(0);
				this.file.getChannel().read(bb);
			} else
			{
				ByteBuffer bb = ByteBuffer.wrap(buffer);
				this.file.getChannel().read(bb);
			}
		} catch (IOException e)
		{
			e.printStackTrace();
		}

		return buffer;
	}

	public RandomAccessFile getFile()
	{
		return file;
	}
}
