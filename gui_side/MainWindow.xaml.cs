using DirectShowLib;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace MotionSense
{
    public static class Globals
    {
        public static string APPS_PATH = Environment.CurrentDirectory + "\\BackProc\\getApps\\apps.txt"; //the file contains suggested apps for the user to define
        public static string DEFINITIONS_PATH = Environment.CurrentDirectory + "\\BackProc\\definitions.json"; //the file contains the defintions the user already defined
        public static string SETTINGS_PATH = Environment.CurrentDirectory + "\\BackProc\\settings.txt"; //the file contains the current state of the switchs
    }

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary
    public partial class MainWindow : Window
    {
        bool powerOn = false; 
        Button currPressed = null; //state of power button

        List<string> nameList = new List<string>();
        List<string> pathList = new List<string>();
        List<ImageSource> iconsList = new List<ImageSource>();
        
        //the function creates the main window
        public MainWindow()
        {
            this.WindowStyle = System.Windows.WindowStyle.None;
            InitializeComponent();

            if (!File.Exists(Globals.DEFINITIONS_PATH))
            {
                using (File.Create(Globals.DEFINITIONS_PATH)) { }
                File.WriteAllText(Globals.DEFINITIONS_PATH, "{}");
            }

            if (!File.Exists(Globals.SETTINGS_PATH))
            {
                using (File.Create(Globals.SETTINGS_PATH)) { }
                string createText = "cam-0\n,pred-0";
                File.WriteAllText(Globals.SETTINGS_PATH, createText);
            }
            else
            {
                string text = File.ReadAllText(Globals.SETTINGS_PATH);
                string cam = text.Split(',')[0];
                string pred = text.Split(',')[1];
                camera_buttonOn = ((int)cam[cam.Length - 1] - '0') != 0;
                predictions_buttonOn = ((int)pred[pred.Length - 1] - '0') != 0;
                if (camera_buttonOn)
                {
                    cameraSlide.IsChecked = true;
                }
                if (predictions_buttonOn)
                {
                    predictSlide.IsChecked = true;
                }
            }

            Process p = new Process();
            p.StartInfo = new ProcessStartInfo("getApps.exe");//running getApps.exe as a process in order to get an up to date list of apps in the users computer 
            p.StartInfo.WorkingDirectory = @"BackProc\getApps";
            p.StartInfo.CreateNoWindow = true;
            p.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
            p.Start();

            Process[] processes = Process.GetProcessesByName("gestureRecorgnition");
            if (processes.Length > 0)//check if the gesture recognition process is currently working
            {
                predictSlide.IsHitTestVisible = false;
                cameraSlide.IsHitTestVisible = false;
                powerImg.Source = (BitmapImage)FindResource("powerOnImgSrc");
                powerOn = true;
            }

            SetAddWindowVars();


            this.contentControl.Content = new AppsWindow(nameList, pathList, iconsList); //create apps window and show it
            appsButton.Style = (Style)FindResource("pressedMenu");
            currPressed = appsButton;
        }

        //the function prepare the varables that are needed for the add window
        public void SetAddWindowVars()
        {
            while (Process.GetProcessesByName("getApps").Length > 0)
            {
                continue;
            }
            Debug.WriteLine("5");
            string[] appsLines = System.IO.File.ReadAllLines(Globals.APPS_PATH);
            int count = 0;
            foreach (string line in appsLines)
            {
                if (count % 2 == 0)
                {
                    nameList.Add(line);
                }
                else
                {
                    try
                    {
                        //Icon icon = System.Drawing.Icon.ExtractAssociatedIcon(line);
                        //iconsList.Add(icon.ToImageSource());
                        pathList.Add(line);
                    }
                    catch
                    {
                        nameList.RemoveAt(nameList.Count - 1);
                    }
                }
                count++;
            }
        }


        //the function takes care of the change in the looks of the window which resulted from the mouse movement
        private void Window_MouseMove(object sender, MouseEventArgs e)
        {
            if (Mouse.LeftButton == MouseButtonState.Pressed && (title.IsMouseOver || titleBar.IsMouseOver || iconImg.IsMouseOver))
            {
                this.DragMove();
            }
        }   

        //the function manages the case that the user pressed on the gui close button
        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            Application.Current.Shutdown();
        }

        //the function manages the case that the user pressed on the gui minimize button
        private void MinimizeButton_Click(object sender, RoutedEventArgs e)
        {
            this.WindowState = System.Windows.WindowState.Minimized;
        }


        //the function manages the case that the user pressed on the process power button
        private void powerClicked(object sender, RoutedEventArgs e)
        {
            if (!powerOn)//if process is off then start process
            {
                DsDevice[] videoInputDevices = DsDevice.GetDevicesOfCat(FilterCategory.VideoInputDevice);
                List<DsDevice> AvailableVideoInputDevices = new List<DsDevice>();
                AvailableVideoInputDevices.AddRange(videoInputDevices);
                int numberOfDevices = AvailableVideoInputDevices.Count;
                if (numberOfDevices > 0)//if camera avaliable start process
                {
                    predictSlide.IsHitTestVisible = false;
                    cameraSlide.IsHitTestVisible = false;
                    Process p = new Process();
                    p.StartInfo = new ProcessStartInfo("runProc.bat");
                    p.StartInfo.WorkingDirectory = @"BackProc";
                    p.StartInfo.CreateNoWindow = true;
                    p.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
                    p.Start();
                    powerImg.Source = (BitmapImage)FindResource("powerOnImgSrc");
                    powerOn = true;
                }
                else//display a window with a warning that there is no camera connected
                {
                    NoCameraPopUp m = new NoCameraPopUp("launch", "No camera connected");
                    m.ShowDialog();
                }
            }
            else//if process is on then stop it
            {
                predictSlide.IsHitTestVisible = true;
                cameraSlide.IsHitTestVisible = true;
                Process[] processes = Process.GetProcessesByName("gestureRecorgnition");
                foreach (var proc in processes)
                {
                    proc.Kill();
                }

                powerImg.Source = (BitmapImage)FindResource("powerOffImgSrc");
                powerOn = false;
            }
        }

        //the function manages the case that the user choose to move to the add window
        private void AddPageClick(object sender, RoutedEventArgs e)
        {
            if (currPressed != (sender as Button))
            {
                this.contentControl.Content = new AddWindow(nameList, pathList, iconsList);
                if (currPressed != null)
                {
                    currPressed.Style = (Style)FindResource("menu");
                }
                (sender as Button).Style = (Style)FindResource("pressedMenu");
                currPressed = (sender as Button);
            }
        }

        //the function manages the case that the user choose to move to the apps window
        private void AppsPageClick(object sender, RoutedEventArgs e)
        {
            this.contentControl.Content = new AppsWindow(nameList, pathList, iconsList);
            if (currPressed != null)
            {
                currPressed.Style = (Style)FindResource("menu");
            }
            (sender as Button).Style = (Style)FindResource("pressedMenu");
            currPressed = (sender as Button);
        }

        //the function checks the state of the camera switch in case the prediction switch is on and write it to "settings.txt"
        public bool predictions_buttonOn = false;
        public void predictions_Checked(object sender, RoutedEventArgs e)
        {
            string createText = "";
            predictions_buttonOn = true;
            if (camera_buttonOn)
            {
                createText = "cam-1,pred-1";
            }
            else
            {
                createText = "cam-0,pred-1";
            }
            File.WriteAllText(Globals.SETTINGS_PATH, createText);
        }

        //the function checks the state of the camera switch in case the prediction switch is off and write it to "settings.txt"
        public void predictions_Unchecked(object sender, RoutedEventArgs e)
        {
            string createText = "";
            predictions_buttonOn = false;
            if (camera_buttonOn)
            {
                createText = "cam-1,pred-0";
            }
            else
            {
                createText = "cam-0,pred-0";
            }
            File.WriteAllText(Globals.SETTINGS_PATH, createText);
        }

        //the function checks the state of the prediction switch in case the camera switch is on and write it to "settings.txt"
        public bool camera_buttonOn = false;
        public void camera_Checked(object sender, RoutedEventArgs e)
        {
            string createText = "";
            camera_buttonOn = true;
            if (predictions_buttonOn)
            {
                createText = "cam-1,pred-1";
            }
            else
            {
                createText = "cam-1,pred-0";
            }
            File.WriteAllText(Globals.SETTINGS_PATH, createText);
        }

        //the function checks the state of the prediction switch in case the camera switch is off and write it to "settings.txt"
        public void camera_Unchecked(object sender, RoutedEventArgs e)
        {
            string createText = "";
            camera_buttonOn = false;
            if (predictions_buttonOn)
            {
                createText = "cam-0,pred-1";
            }
            else
            {
                createText = "cam-0,pred-0";
            }
            File.WriteAllText(Globals.SETTINGS_PATH, createText);
        }
    }
}