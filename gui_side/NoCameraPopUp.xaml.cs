using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace MotionSense
{
    /// <summary>
    /// Interaction logic for NoCameraPopUp.xaml
    /// </summary>
    
    public partial class NoCameraPopUp : Window
    {
        //the function checks if there isn't any camera connected to the computer and if there isn't, the function displays a window with a warning
        public NoCameraPopUp(string topic, string msg)
        {
            InitializeComponent();

            TextBlock text = new TextBlock();
            text.Text = msg;
            Topic.Content = "Unable to " + topic;
            text.HorizontalAlignment = HorizontalAlignment.Center;
            text.VerticalAlignment = VerticalAlignment.Center;
            DynamicGrid.Children.Add(text);

            Grid.SetRow(text, 1);
        }

        //the function closes the warning window when the ok button is pressed by the user
        private void OK_Button_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }

    }
}
