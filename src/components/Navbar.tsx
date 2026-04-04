import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { Scan, Menu, X, Sun, Moon } from "lucide-react";
import { useState, useEffect } from "react";

const navItems: { label: string; path: string }[] = [];

const Navbar = () => {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isLight, setIsLight] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle("light", isLight);
  }, [isLight]);

  return (
    <motion.nav
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="fixed top-0 left-0 right-0 z-50 glass-strong"
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" onClick={() => { window.location.href = '/'; }} className="flex items-center gap-2 group">
          <div className="w-8 h-8 rounded-lg overflow-hidden flex items-center justify-center">
            <img src="https://i.ibb.co/ymbx436j/images-Photoroom.png" alt="RoadSense logo" className="w-full h-full object-contain" />
          </div>
          <span className="font-display font-bold text-lg text-foreground">
            Road<span className="text-primary">Sense</span>
          </span>
        </Link>

        <div className="flex items-center gap-1">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                location.pathname === item.path
                  ? "text-primary bg-primary/10"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary"
              }`}
            >
              {item.label}
            </Link>
          ))}
          <button
            onClick={() => setIsLight((v) => !v)}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
            title={isLight ? "Switch to dark" : "Switch to light"}
          >
            {isLight ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
          </button>
          <button
            className="md:hidden text-foreground ml-1"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {mobileOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="md:hidden glass-strong border-t border-border px-6 py-4 space-y-2"
        >
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              onClick={() => setMobileOpen(false)}
              className={`block px-4 py-2 rounded-lg text-sm font-medium ${
                location.pathname === item.path
                  ? "text-primary bg-primary/10"
                  : "text-muted-foreground"
              }`}
            >
              {item.label}
            </Link>
          ))}
        </motion.div>
      )}
    </motion.nav>
  );
};

export default Navbar;
