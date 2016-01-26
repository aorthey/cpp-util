#include <gtkmm/main.h>
#include <gtkmm/window.h>
#include <gtkmm/button.h>
#include <iostream>
#include <vector>



class MainWindow : public Gtk::Window
{

public:
  MainWindow(const char *str):
        m_button(str),
 
    set_title( "MainWindow" );
    //set_border_width(12);



    m_button.signal_clicked().connect(
        sigc::mem_fun( *this, &MainWindow::on_button_clicked ) );



    show_all_children();
  }
  virtual ~MainWindow(){};

protected:

  void on_button_clicked(){
    std::cout << "Hello world" << std::endl;
  }

  Gtk::Button m_button;
};
int main(int argc, char *argv[])
{


  Gtk::Main kit(argc, argv);
  MainWindow m("NoChange");

  Gtk::Main::run(m);

  return 0;
}
