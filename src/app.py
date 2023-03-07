# include dependencies
from image_jam_gui import ImageJamGUI


# main function
def main() -> None:
    """Main function"""
    # create the gui object
    gui = ImageJamGUI()
    # run the gui
    gui.run()


if __name__ == "__main__":
    main()
