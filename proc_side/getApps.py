import sys
import win32com.client
import os
import re
import os
import configparser

def get_apps_from_dir(dir):
	regex = "^['a-zA-Z0-9 +-.&]*$"
	shell = win32com.client.Dispatch("WScript.Shell")
	apps_names = {}
	for root, dirs, files in os.walk(dir):
		for name in files:
			full_path = os.path.join(root, name)
			lower_name = name.lower()
			if len(re.findall(regex, name)) > 0: 
				if lower_name.split(".")[-1] == "lnk":
					first_word = lower_name.split(' ')[0]
					if "uninstall" not in lower_name and "x64" not in lower_name and "x86" not in lower_name and "download" not in first_word and "install" not in first_word:
						target_path = shell.CreateShortCut(full_path).Targetpath
						if os.path.exists(target_path):
							if target_path.split(".")[-1].lower() == "exe" and name.lower().split(".lnk")[0] not in apps_names.keys():
								apps_names[name.split(".lnk")[0]] = target_path
				elif lower_name.split(".")[-1] == "url":
					config = configparser.RawConfigParser()
					config.read(full_path)

					try:
						url = config.get('InternetShortcut', 'URL')  # Classic URL Format
					except configparser.NoOptionError:
						url = config.get('DEFAULT', 'BASEURL')  # Extended URL Format
					if "http" not in url:	
						#print(url)
						apps_names[name.split(".url")[0]] = full_path

	return apps_names
	

def main():	
	dir1 = "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs"
	dir2 = "C:" + os.path.join(os.environ["HOMEPATH"], "Desktop")
	
	apps1 = get_apps_from_dir(dir1)
	apps2 = get_apps_from_dir(dir2)
	apps = {}
	for name, path in apps1.items():
		if path not in apps2.values():	
			apps[name] = path
			
	apps.update(apps2)
	
	file = open("apps.txt", "w")
	
	for name, path in apps.items():
		duplicate = False
		filename = path.split("\\")[-1]
		for name1, path1 in apps.items(): 
			filename1 = path1.split("\\")[-1]
			if not (name1 == name and path1 == path):
				if filename == filename1 and name1 in name:
					duplicate = True
					break
		
		if not duplicate:
			#print(name)
			#print(filename)
			#print()
			file.write(name + "\n")
			file.write(path + "\n")	


main() 