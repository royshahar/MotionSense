using Microsoft.Win32;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
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
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MotionSense
{
    /// <summary>
    /// Interaction logic for AddWindow.xaml
    /// </summary>
    public partial class AddWindow : UserControl
    {
        List<string> nameList = new List<string>();
        List<string> pathList = new List<string>();
        List<string> addedNames = new List<string>();
        List<ImageSource> iconsList = new List<ImageSource>();

        string[] suggestedApps = { "powerpoint", "powerpoint 2013", "powerpoint 2016", "word", "word 2013", "word 2016", "excel", "excel 2013", "excel 2016", "paint", "wordpad",
                                    "windows media player", "vlc media player", "unity", "irfanview 64", "skype", "discord", "google chrome", "paint.net", "blender", "netflix",
                                    "steam", "plex", "firefox", "plex media player", "fortnite", "wireshark", "origin", "popcorn time", "visual studio 2015", "visual studio 2017",
                                    "skype for business", "teamviewer 13", "android studio", "jetbrains pycharm 2018.2.1", "apex legends", "dosbox 0.74", "battlefield 1",
                                    "call of duty black ops 4", "fifa 17", "fifa 18", "fifa 19", "nba 2k17", "nba 2k18", "nba 2k19", "hxd"};

        //The function create the add window 
        public AddWindow(List<string> nameList1, List<string> pathList1, List<ImageSource> iconsList1)
        {
            InitializeComponent();

            nameList = nameList1;
            pathList = pathList1;
            iconsList = iconsList1;

            DefinedNames();
            LoadImages();

            txtAuto.TextChanged += new TextChangedEventHandler(txtAuto_TextChanged);
        }

        //the function manage the search bar
        #region TextBox-TextChanged-txtAuto
        private void txtAuto_TextChanged(object sender, TextChangedEventArgs e)
        {
            string NormaltypedString = txtAuto.Text;
            string typedString = NormaltypedString.ToLower();
            List<ListApp> listApps = new List<ListApp>();
            bool found = false;
            listApps.Clear();
            int itemCounter = 0;

            if (!nameList.Contains(NormaltypedString))
            {
                foreach (string item in nameList)
                {
                    string lowerItem = item.ToLower();
                    if (!string.IsNullOrEmpty(typedString))
                    {
                        found = false;
                        if (lowerItem.StartsWith(typedString))
                        {
                            found = true;
                        }
                        else
                        {
                            foreach (string word in lowerItem.Split(' '))
                            {
                                if (word.StartsWith(typedString))
                                {
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if (found && itemCounter < 4 && !addedNames.Contains(item))
                        {
                            try
                            {
                                ImageSource icon = Icon.ExtractAssociatedIcon(pathList[nameList.IndexOf(item)]).ToImageSource();
                                listApps.Add(new ListApp(item, icon));
                                //listApps.Add(new ListApp(item, iconsList[nameList.IndexOf(item)]));
                                itemCounter++;
                            }
                            catch
                            {
                                Console.WriteLine("Couldn't get icon");
                            }
                        }
                    }
                }
            }

            if (listApps.Count > 0)
            {
                lbSuggestion.ItemsSource = listApps;
                lbSuggestion.Visibility = Visibility.Visible;
            }
            else if (txtAuto.Text.Equals(""))
            {
                lbSuggestion.Visibility = Visibility.Collapsed;
                lbSuggestion.ItemsSource = null;
            }
            else
            {
                lbSuggestion.Visibility = Visibility.Collapsed;
                lbSuggestion.ItemsSource = null;
            }
        }
        #endregion

        #region ListBox-SelectionChanged-lbSuggestion
        private void lbSuggestion_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (lbSuggestion.ItemsSource != null)
            {
                lbSuggestion.Visibility = Visibility.Collapsed;
                txtAuto.TextChanged -= new TextChangedEventHandler(txtAuto_TextChanged);
                if (lbSuggestion.SelectedIndex != -1)
                {
                    txtAuto.Text = ((ListApp)lbSuggestion.SelectedItem).appName;
                    txtAuto.BorderBrush = System.Windows.Media.Brushes.Gray;
                    lbSuggestion.SelectedIndex = -1;
                }
                txtAuto.TextChanged += new TextChangedEventHandler(txtAuto_TextChanged);
            }
        }
        #endregion

        //the function loads the icons of the application that are suggested to the user
        public void LoadImages()
        {
            List<ListApp> listApps = new List<ListApp>();

            int counter = 0;
            foreach (string app in suggestedApps)
            {
                List<string> myList = nameList.ConvertAll(d => d.ToLower());
                if (myList.Contains(app) && counter < 10 && !addedNames.Contains(app))
                {
                    ImageSource icon;
                    try
                    {
                        icon = Icon.ExtractAssociatedIcon(pathList[myList.IndexOf(app)]).ToImageSource();
                        //ImageSource icon = iconsList[myList.IndexOf(app)];
                    }
                    catch
                    {
                        Console.WriteLine("Couldn't get icon");
                        continue;
                    }
                    string name = nameList[myList.IndexOf(app)];
                    if (name.Length > 16)
                    {
                        name = name.Substring(0, 13);
                        name += "...";
                    }
                    if (addedNames == null || !(addedNames.Contains(name)))
                    {
                        listApps.Add(new ListApp(name, icon));
                        counter++;
                    }
                }
            }
            IconListView.ItemsSource = listApps;
        }


        //the function define the file explorer in which the user can browse for an app to define
        private void btnOpenFiles_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Exe files (*.exe)|*.exe";
            openFileDialog.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            if (openFileDialog.ShowDialog() == true)
            {
                txtAuto.Text = openFileDialog.FileName;
                txtAuto.BorderBrush = System.Windows.Media.Brushes.Gray;
            }
        }

        //the function creates the define button
        private void DefineButton_Click(object sender, RoutedEventArgs e)
        {
            if (File.Exists(txtAuto.Text))
            {
                string[] appsLines = System.IO.File.ReadAllLines(Globals.APPS_PATH);
                int count = 0;
                string name = "";
                bool inApps = false;
                foreach (string line in appsLines)
                {
                    if (count % 2 == 0)
                    {
                        name = line;
                    }
                    else
                    {
                        if (line == txtAuto.Text)
                        {
                            inApps = true;
                            break;
                        }
                    }
                    count++;
                }
                if (inApps)
                {
                    int index = nameList.IndexOf(name);
                    ImageSource icon = Icon.ExtractAssociatedIcon(pathList[index]).ToImageSource();
                    //DefineWindow window1 = new DefineWindow(name, iconsList[index], "");
                    DefineWindow2 window1 = new DefineWindow2(name, icon, "");
                    window1.ShowDialog();
                    if (window1.saveButton.Content.ToString() == "1")
                    {
                        Console.WriteLine("1");
                        DefinedNames();
                        LoadImages();
                    }
                    txtAuto.Text = "";
                }
                else
                {
                    string path = txtAuto.Text;
                    Icon icon = Icon.ExtractAssociatedIcon(path);
                    string appName = System.IO.Path.GetFileNameWithoutExtension(path);
                    DefineWindow2 window1 = new DefineWindow2(appName, icon.ToImageSource(), txtAuto.Text);
                    window1.ShowDialog();
                    if (window1.saveButton.Content.ToString() == "1")
                    {
                        Console.WriteLine("1");
                        DefinedNames();
                        LoadImages();
                    }
                    txtAuto.Text = "";
                    txtAuto.BorderBrush = System.Windows.Media.Brushes.Gray;
                }
                
            }
            else if (nameList.Contains(txtAuto.Text))
            {
                int index = nameList.IndexOf(txtAuto.Text);
                ImageSource icon = Icon.ExtractAssociatedIcon(pathList[index]).ToImageSource();
                //DefineWindow window1 = new DefineWindow(txtAuto.Text, iconsList[index], "");
                DefineWindow2 window1 = new DefineWindow2(txtAuto.Text, icon, "");
                window1.ShowDialog();
                if (window1.saveButton.Content.ToString() == "1")
                {
                    Console.WriteLine("1");
                    DefinedNames();
                    LoadImages();
                }
                txtAuto.Text = "";
                txtAuto.BorderBrush = System.Windows.Media.Brushes.Gray;
            }
            else
            {
                txtAuto.BorderBrush = System.Windows.Media.Brushes.Red;
            }
        }

        //the function manages a case when the user pressed on an app's icon
        private void SuggestedPressed(object sender, SelectionChangedEventArgs e)
        {
            if (IconListView.SelectedItem != null)
            {
                string appName = ((ListApp)IconListView.SelectedItem).appName;
                ImageSource appIcon = ((ListApp)IconListView.SelectedItem).iconImg;
                Console.WriteLine(appName);
                DefineWindow2 window1 = new DefineWindow2(appName, appIcon, "");
                window1.ShowDialog();
                txtAuto.BorderBrush = System.Windows.Media.Brushes.Gray;
                if (window1.saveButton.Content.ToString() == "1")
                {
                    Console.WriteLine("1");
                    DefinedNames();
                    LoadImages();
                }
                IconListView.SelectedItem = null;
            }
        }

        //the function updates the dictionary that the add window use in order to know which apps are defined
        public void DefinedNames()
        {
            Dictionary<string, Dictionary<string, string>> previousDictionary = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, string>>>(System.IO.File.ReadAllText(Globals.DEFINITIONS_PATH));
            List<ListApp> listApps = new List<ListApp>();
            addedNames.Clear();
            if (previousDictionary != null)
            {
                foreach (KeyValuePair<string, Dictionary<string, string>> entry in previousDictionary)
                {
                    addedNames.Add(entry.Key.Split('~')[0]);
                }
            }                
        }
    }
}
