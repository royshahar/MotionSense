using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
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
    /// Interaction logic for AppsWindow.xaml
    /// </summary>
    //the class define the apps window logics
    public partial class AppsWindow : UserControl
    {
        List<string> nameList = new List<string>();
        List<string> pathList = new List<string>();
        List<ImageSource> iconsList = new List<ImageSource>();
        List<string> definesApps = new List<string>();

        //init function, creates the app window
        public AppsWindow(List<string> nameList1, List<string> pathList1, List<ImageSource> iconsList1)
        {
            InitializeComponent();

            nameList = nameList1;
            pathList = pathList1;
            iconsList = iconsList1;

            ReadDefinitions();
        }

        //the function handels the case that the user pressed on one of the apps icons in the window, it opens a defintions window for the app
        private void SuggestedPressed(object sender, SelectionChangedEventArgs e)
        {
            if (IconListView.SelectedItem != null)
            {
                string appName = ((ListApp)IconListView.SelectedItem).appName;
                if (appName.EndsWith("..."))
                {
                    appName = appName.Substring(0, appName.Length - 3);

                    foreach (string item in nameList)
                    {
                        string sub = item;
                        if (sub.Length > 18)
                        {
                            sub = sub.Substring(0, 16);
                        }
                        if (sub == appName)
                        {
                            appName = item;
                        }
                    }
                }
                ImageSource appIcon = ((ListApp)IconListView.SelectedItem).iconImg;
                Console.WriteLine(appName);
                DefineWindow2 window1 = new DefineWindow2(appName, appIcon, "");
                window1.ShowDialog();
                IconListView.SelectedItem = null;
            }
        }

        //the function reads the defintions the user defined in the past from defintions.json and display the names and icons of the apps that are already defined in the window
        public void ReadDefinitions()
        {
            Dictionary<string, Dictionary<string, string>> previousDictionary = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, string>>>(System.IO.File.ReadAllText(Globals.DEFINITIONS_PATH));
            List<ListApp> listApps = new List<ListApp>();
            if (previousDictionary != null)
            {
                foreach (KeyValuePair<string, Dictionary<string, string>> entry in previousDictionary)
                {
                    Dictionary<string, string> innerDict = entry.Value;
                    string key = entry.Key.Split('~')[0];

                    if (nameList.Contains(key))
                    {
                        ImageSource icon;
                        try
                        {
                            icon = Icon.ExtractAssociatedIcon(pathList[nameList.IndexOf(key)]).ToImageSource();
                            //ImageSource icon = iconsList[nameList.IndexOf(key)];
                        }
                        catch
                        {
                            Console.WriteLine("Couldn't get icon");
                            continue;
                        }
                        string name = key;
                        if (name.Length > 18)
                        {
                            name = name.Substring(0, 16);
                            name += "...";
                        }
                        listApps.Add(new ListApp(name, icon));
                    }
                    else if (File.Exists(entry.Key.Split('~')[1]))
                    {
                        Icon icon = Icon.ExtractAssociatedIcon(entry.Key.Split('~')[1]);
                        string name = key;
                        if (name.Length > 18)
                        {
                            name = name.Substring(0, 16);
                            name += "...";
                        }
                        listApps.Add(new ListApp(name, icon.ToImageSource()));
                    }
                }
            }

            List<ListApp> SortedList = listApps.OrderBy(o => o.appName).ToList();
            Uri oUri = new Uri("pack://application:,,,/" + Assembly.GetExecutingAssembly().GetName().Name + ";component/Images/default3.png", UriKind.RelativeOrAbsolute);
            Console.WriteLine(oUri.AbsolutePath);
            BitmapImage img = new BitmapImage(oUri);
            ListApp defaultDef = new ListApp("Default", img);
            SortedList.Add(defaultDef);

            IconListView.ItemsSource = SortedList;
        }

        //the function handels the case that a user pressed the 'x' sign above one of the application icons which means that he decided to delete this app's definitions
        public void DeleteDefinition(object sender, RoutedEventArgs e)
        {
            string appName = (sender as Button).Tag.ToString();
            Dictionary<string, Dictionary<string, string>> previousDictionary = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, string>>>(System.IO.File.ReadAllText(Globals.DEFINITIONS_PATH));

            if (appName.EndsWith("..."))
            {
                appName = appName.Substring(0, appName.Length - 3);

                foreach (string item in nameList)
                {
                    string sub = item;
                    if (sub.Length > 18)
                    {
                        sub = sub.Substring(0, 16);
                    }
                    if (sub == appName)
                    {
                        appName = item;
                    }
                }
            }

            foreach (string entry in previousDictionary.Keys)
            {
                if (entry.Split('~')[0] == appName)
                {
                    previousDictionary.Remove(entry);
                    break;
                }
            }

            string json = JsonConvert.SerializeObject(previousDictionary);
            System.IO.File.WriteAllText(Globals.DEFINITIONS_PATH, json);
            ReadDefinitions();
        }
    }
}