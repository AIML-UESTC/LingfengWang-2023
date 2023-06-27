### Structure
Here are each of the main folders/files and what they do.

- `/apps` - This is where we have an `app_<name>.py` file for each app within our overall streamlit app. For example, `/apps/app_silly_strings.py` contains as much of the logic and code that is specific to the "silly string stuff" app.
- `/assets` - The place where your images and other similar types of files live that you want to use across your apps.
- `/src` - For python code and functions you want to have available on any page of your app you can make a custom module in here. For example the `funny_numbers` module has the functions used by the "crazy numbers" app.
- `app.py` - This is the main streamlit app we launch via `streamlit run app.py` which will import and render the other apps as needed.
- `/resources` - Contain resources files（images、videos）
- `/ckpts` - Contain model weights
- `/wheel` - Contain pytorch-cuda10.2 building wheels

### Get started
1. Download resources、ckpts、wheel folder from BaiduNetDisk（Password：tars） [https://pan.baidu.com/s/1UIf1vAJaa6T8UEl7DGqnLA ](https://pan.baidu.com/s/1UIf1vAJaa6T8UEl7DGqnLA )
2. Install pytorch wheel and other required python packages
3. Excute `Streamlit run app.py`

This branch only provide inference code.
Training code is provided in master branch.