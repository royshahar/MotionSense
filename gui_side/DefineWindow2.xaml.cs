using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace MotionSense
{
    /// <summary>
    /// Interaction logic for Window1.xaml
    /// </summary>
    //the class creates define app window for seven gestures
    public partial class DefineWindow2 : Window
    {
        //dictionary that define specail keys in the keyboard
        public Dictionary<Key, string> dict = new Dictionary<Key, string>()
        {
            { Key.Back, "Backspace" }, { Key.Escape, "Esc" }, { Key.Oem3, "`" }, { Key.Return, "Enter" }, { Key.OemMinus, "-" }, { Key.OemPlus, "=" }, { Key.OemOpenBrackets, "[" },
            { Key.Oem6, "]" }, { Key.Oem5, "\\" }, { Key.Oem1, ";" }, { Key.OemQuotes, "'" }, { Key.OemComma, "," }, { Key.OemPeriod, "." }, { Key.OemQuestion, "/" }
        };

        //init function creates a define app window
        public string browsePath = "";
        public DefineWindow2(string appName, ImageSource iconImg, string browsePathArg)
        {
            InitializeComponent();

            if (browsePathArg != "")
            {
                browsePath = browsePathArg;
            }

            appDefineName.Text = appName;
            iconDefineImg.Source = iconImg;

            defineHeadline.Width = iconDefineImg.Width + appDefineName.Width + 20;

            foreach (Button tb in grid.Children.OfType<Button>())
            {
                if (tb.Name != "exitBtn" && tb.Name != "minBtn" && tb.Name != "saveButton" && tb.Name != "cancelButton")
                {
                    tb.Content = "???";
                    tb.Click += Button_Click;
                    tb.FontSize = 15;
                    tb.FontWeight = FontWeights.Medium;
                }
            }

            Dictionary<string, Dictionary<string, string>> previousDictionary = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, string>>>(System.IO.File.ReadAllText(Globals.DEFINITIONS_PATH));
            if (previousDictionary != null)
            {
                foreach (KeyValuePair<string, Dictionary<string, string>> entry in previousDictionary)
                {
                    if (entry.Key.Split('~')[0] == appName)
                    {
                        Dictionary<string, string> innerDict = entry.Value;
                        foreach (KeyValuePair<string, string> innerEntry in innerDict)
                        {
                            foreach (Button tb in grid.Children.OfType<Button>())
                            {
                                string gesture = innerEntry.Key.Replace(' ', '_');
                                if (tb.Name == gesture)
                                {
                                    tb.Content = innerEntry.Value;
                                    tb.Foreground = Brushes.Blue;
                                }
                            }
                        }
                    }
                }
            }

            this.PreviewKeyDown += new KeyEventHandler(OnPreviewKeyDown);
        }

        [DllImport("user32.dll", CharSet = CharSet.Auto, ExactSpelling = true, CallingConvention = CallingConvention.Winapi)]
        public static extern short GetKeyState(int keyCode);//take key state fron the os

        //the function fixes the input key from the user to be a lower key or a combination that is defined by the system
        public void OnPreviewKeyDown(object sender, KeyEventArgs e)
        {
            string pressed = e.Key.ToString();
            Console.WriteLine("prev: " + pressed);
            if (e.Key == Key.LeftShift || e.Key == Key.RightShift || e.Key == Key.RightCtrl || e.Key == Key.LeftCtrl || e.Key == Key.System)
            {
                Console.WriteLine("ret");
                return;
            }
            if (dict.ContainsKey(e.Key))
            {
                pressed = dict[e.Key];
            }
            else if (e.Key >= Key.A && e.Key <= Key.Z)
            {
                bool CapsLock = (((ushort)GetKeyState(0x14)) & 0xffff) != 0;
                if (!CapsLock)
                {
                    pressed = e.Key.ToString().ToLower();
                }
            }
            else if (pressed.Any(char.IsDigit) && pressed[0] != 'F')
            {
                pressed = pressed[pressed.Length - 1].ToString();
            }

            if ((Keyboard.IsKeyDown(Key.LeftCtrl) || Keyboard.IsKeyDown(Key.RightCtrl)) && (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift)))
            {
                pressed = "Ctrl+Shift+" + pressed;
            }
            else if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift))
            {
                pressed = "Shift+" + pressed;
                Console.WriteLine(pressed);
            }
            else if (Keyboard.IsKeyDown(Key.LeftCtrl) || Keyboard.IsKeyDown(Key.RightCtrl))
            {
                pressed = "Ctrl+" + pressed;
            }

            foreach (Button tb in grid.Children.OfType<Button>())
            {
                if (tb.Name != "exitBtn" && tb.Name != "minBtn" && tb.Name != "saveButton" && tb.Name != "cancelButton")
                {
                    if (tb.Content.ToString() == ">???<")
                    {
                        if (pressed.Length > 11)
                        {
                            tb.FontSize = tb.FontSize - (int)((pressed.Length - 11) * 0.7);
                        }
                        tb.Content = pressed;
                        Console.WriteLine(pressed);
                        tb.Foreground = Brushes.Blue;
                    }
                }
            }
            e.Handled = true;
        }

        //the function manages the case that the user click on one of the buttons in the define window
        public void Button_Click(object sender, RoutedEventArgs e)
        {
            if ((sender as Button).Content.ToString() == ">???<")
            {
                (sender as Button).Content = "???";
                (sender as Button).Foreground = Brushes.Black;
            }
            else
            {
                foreach (Button tb in grid.Children.OfType<Button>())
                {
                    if (tb.Name != "exitBtn" && tb.Name != "minBtn" && tb.Name != "saveButton" && tb.Name != "cancelButton")
                    {
                        if (tb.Content.ToString() == ">???<")
                        {
                            tb.Content = "???";
                            tb.FontSize = 15;
                            tb.Foreground = Brushes.Black;
                        }
                    }
                    (sender as Button).Content = ">???<";
                    (sender as Button).Foreground = Brushes.Red;
                }
            }
        }

        //the function takes care of the change in the looks of the window that is resulted from the mouse movements
        private void Window_MouseMove(object sender, MouseEventArgs e)
        {
            if (Mouse.LeftButton == MouseButtonState.Pressed && (title.IsMouseOver || titleBar.IsMouseOver || iconImg.IsMouseOver))
            {
                this.DragMove();
            }
        }

        //the function manages the case that the user clicked on the gui close button
        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }

        //the function manages the case that the user clicked on the gui minimize button
        private void MinimizeButton_Click(object sender, RoutedEventArgs e)
        {
            this.WindowState = WindowState.Minimized;
        }

        //the function manages the case that the user clicked on the save definitions button
        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            Dictionary<string, Dictionary<string, string>> previousDictionary = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, string>>>(System.IO.File.ReadAllText(Globals.DEFINITIONS_PATH));
            Dictionary<string, string> innerDict = new Dictionary<string, string>();
            foreach (Button tb in grid.Children.OfType<Button>())
            {
                if (tb.Name != "exitBtn" && tb.Name != "minBtn" && tb.Name != "saveButton" && tb.Name != "cancelButton")
                {
                    if (tb.Content.ToString() != "???" && tb.Content.ToString() != ">???<")
                    {
                        string gesture = tb.Name.Replace('_', ' ');
                        innerDict.Add(gesture, tb.Content.ToString());
                    }
                }
            }
            string app = appDefineName.Text + "~";
            string[] appsLines = System.IO.File.ReadAllLines(Globals.APPS_PATH);
            bool found = false;
            foreach (string line in appsLines)
            {
                if (line == appDefineName.Text)
                {
                    found = true;
                }
                else if (found)
                {
                    app += line;
                    break;
                }
            }
            if (browsePath != "")
            {
                app += browsePath;
            }

            if (previousDictionary == null)
            {
                previousDictionary = new Dictionary<string, Dictionary<string, string>>();
            }
            else if (previousDictionary.ContainsKey(app))
            {
                previousDictionary.Remove(app);
            }
            previousDictionary.Add(app, innerDict);

            string json = JsonConvert.SerializeObject(previousDictionary);
            if (innerDict.Count > 0)
            {
                System.IO.File.WriteAllText(Globals.DEFINITIONS_PATH, json);
                saveButton.Content = "1";
            }
            this.Close();
        }

        //the function manages the case that the user clicked on the cancel
        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
    }
}
