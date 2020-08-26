package StructuralElements;

import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;

public class myImage {
	
	int height, weigth;
	BufferedImage bImage;
	
	public myImage(int h, int w)
	{
		this.height = h;
		this.weigth = w;
		
		bImage = new BufferedImage(weigth, height, BufferedImage.TYPE_BYTE_BINARY);
	}
	
	public void drawPolygons(ArrayList<Polygon> polygons)
	{
		// Create a graphics contents on the buffered image
	    Graphics2D g2d = bImage.createGraphics();
	    
		for(int i=0; i<polygons.size();i++)
		{
			g2d.fillPolygon(polygons.get(i));
		}
	}
	
	public void drawPolygons(ArrayList<Polygon> polygons, int negated)
	{
		// Create a graphics contents on the buffered image
	    Graphics2D g2d = bImage.createGraphics();
	    
		for(int i=0; i<polygons.size();i++)
		{
			g2d.fillPolygon(polygons.get(i));
		}
		// negates the image for the written polygons
		if(negated == 1)
		{
			for(int x = 0; x<height; x++)
			{
				for(int y = 0; y<weigth; y++)
				{
					if(bImage.getRGB(y,x)==-16777216)
						bImage.setRGB(y,x, -1);
					else bImage.setRGB(y,x, -16777216);
				}
			}
		}
	}
	
	public BufferedImage getBufferedImage()
	{
		return bImage;
	}
	
	public void writeImage(File file, String type) throws IOException
	{
		ImageIO.write(bImage, type, file);
	}

}
