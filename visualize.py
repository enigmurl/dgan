from manimlib import *
from nn import *
from config import *
from train import training_data

import numpy as np

MAX_POINTS = 64
DY = -1
GY = -1
#not really scalable, quick and easy preview of generator and discriminator
class Train(Scene):
    
    def construct(self):
        data  = training_data(DATA_ELEMENTS)
        data2 = training_data(DATA_ELEMENTS * DGAN_COMPARE_ENTRIES)

        truth, _ = self.tile(-2,DY, "Truth", self.points(0,0,data[:MAX_POINTS])); 
        g, gpoints = self.tile(0,DY, "GAN G", self.points(0,0,[[0.5,0.5]]));
        d, dpoints = self.tile(2,DY, "DGAN G", self.points(0,0,[[0.5,0.5]]));

        gg, ggpoints = self.disc_tile(0,GY, "GAN D",  lambda x, training : [[0]])
        dd, ddpoints = self.disc_tile(2,GY, "DGAN D", lambda x, training : [[0]])

        self.play(ShowCreation(truth),ShowCreation(gg), ShowCreation(dd), ShowCreation(g), ShowCreation(d)) 

        title = Text("Epoch: ").to_corner(UP + LEFT);
        self.add(title);
        number = Text("").next_to(title, RIGHT);
        self.add(number);
         

        for i, (gan, dgan) in enumerate(zip(gan_train(data), dgan_train(data))):
        
            self.play(Transform(gpoints, self.points(0,DY, gan[0][:MAX_POINTS])), 
                      Transform(dpoints, self.points(2,DY, dgan[0][:MAX_POINTS])),
                      Transform(number, Text(str(i)).next_to(title, RIGHT)),
                      Transform(ggpoints, self.sub_tiles(0,GY, gan[1])),
                      Transform(ddpoints, self.sub_tiles(2,GY, dgan[1], DGAN_COMPARE_ENTRIES))
                        );

    def disc_tile(self, x, y, label, func):
        return self.tile(x, y, label, self.sub_tiles(0,0, func))

    def sub_tiles(self,x,y, func, multiply = 1):
        sample_axis = 8;
        mobs = [];
        for r in range(sample_axis + 1):
            for c in range(sample_axis + 1):
                blue = func(tf.constant([[c / sample_axis, r / sample_axis] * multiply]), True)
                color = Color(rgb=interpolate(hex_to_rgb(PURPLE), hex_to_rgb(BLUE), blue[0][0]))
                sq = Square(side_length=1/sample_axis).set_fill(color).set_opacity(0.25).set_stroke(color)
                sq.move_to(1.5 * np.array([c / sample_axis - 0.5, r / sample_axis - 0.5, 0]) + np.array([x,y,0]))        
                mobs.append(sq);
       
        return VGroup(*mobs); 

    def tile(self, x, y, label, points):
        square = Square().set_stroke(RED);
        text = Text(label)

        main = VGroup(square, points);
        main.move_to(np.array([x,y,0]))
        text.next_to(main, DOWN);

        group = VGroup(main, text);
        return group, points;

    def points(self, x,y, points):
        return VGroup(*[SmallDot(
            1.5 * (np.array([*p,0]) - np.array([0.5, 0.5, 0])) + np.array([x,y,0])
                        ) for p in points])
    
