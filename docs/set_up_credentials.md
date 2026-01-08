# Instructions how to generate and enable a google Cloud Vision API key

1. Go to [google-cloud-vision](https://cloud.google.com/vision) and click on "Console". Sign into your google account / create a new google account if prompted. This will bring you to the following page, where you click on "project" in the top of the screen.
![img0](_static/img0.png)
2. Select "project" from the top left drop-down menu.
![img1](_static/img1.png)
3. Click on "NEW PROJECT" on the left of the pop-up window.
![img2](_static/img2.png)
4. Enter a project name and click on "CREATE".
![img3](_static/img3.png)
5. Now you should be back on the dashboard. In the top right, click on the three vertical dots.
![img4](_static/img4.png)
6. In the drop-down menu, select "Project settings".
![img5](_static/img5.png)
7. In the menu on the left, click on "Service Accounts".
![img6](_static/img6.png)
8. Click on "+ CREATE SERVICE ACCOUNT".
![img7](_static/img7.png)
9. Select a service account ID (you can pick this as any name you wish). Click on "DONE".
![img8](_static/img8.png)
10. Now your service account should show up in the list of service accounts.
![img9](_static/img9.png)
11. Click on the three vertical dots to the right of your service account name and select "Manage keys".
![img10](_static/img10.png)
12. Click on "Create new key".
![img11](_static/img11.png)
13. In the pop-up window, select "JSON" and click "CREATE".
![img12](_static/img12.png)
14. The private key is directly downloaded to your computer. It should be in your downloads folder.
![img13](_static/img13.png)
15. The JSON key file will look something like this (any private information has been blanked out in the screenshot).
![img14](_static/img14.png)
16. Now go back to your browser window. Click on "Google Cloud" in the top left corner.
![img15](_static/img15.png)
17. Now select "APIs & Services".
![img16](_static/img16.png)
18. From the selection of APIs, select "Cloud Vision API" or search for it and then select.
![img17](_static/img17.png)
19. Click on "ENABLE".
![img18](_static/img18.png)
20. Google Cloud Vision API is now enabled for your key.
![img19](_static/img19.png)
21. Place the JSON key in a selected folder on your computer and reference this key in your Jupyter Notebook / Python console when running ammico. Or, upload it to your google Drive to use it on google Colaboratory.
22. Make sure that [billing is enabled](https://support.google.com/googleapi/answer/6158867?hl=en) for your google account. You can get the first three month for free; after that, you will be charged if processing more than 1000 images / month (currently $1.50 per 1000 images, see [here](https://cloud.google.com/vision/pricing/)).
