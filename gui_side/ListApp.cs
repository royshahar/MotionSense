using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;

namespace MotionSense
{
    //the class is a container which his values are app's name and the app's icon
    class ListApp
    {
        public string appName { get; set; }
        public ImageSource iconImg { get; set; }

        public ListApp(string name, ImageSource img)
        {
            appName = name;
            iconImg = img;
        }
    }
}
